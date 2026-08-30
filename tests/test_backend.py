from __future__ import annotations

import asyncio
import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from backend.app.config import Settings
from backend.app.events import EventHub
from backend.app.loss_parser import LossParser
from backend.app.models import TrainingConfig
from backend.app.processes import ProcessRun
from backend.app.paths import (
    clear_dataset_directory,
    dataset_counts,
    dataset_directory_for_name,
    gdrive_uri,
    normalize_local_path,
    validate_gdrive_path,
    validate_image_path,
    validate_local_dataset,
)
from backend.app.recommendations import recommend_settings
from backend.app.rclone_sync import RcloneService
from backend.app.summary import representative_step_records, write_training_summary
from backend.app.tag_ops import (
    batch_remove_tags,
    batch_update_tags,
    list_image_records,
    read_tags,
    update_image_tags,
)
from backend.app.training import TrainingService, build_dataset_toml, build_sample_prompts, build_training_command
from backend.bundled_tagger import run_mock_tagger


class BackendContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # The Codex Windows sandbox may deny writes to the system TEMP folder;
        # keeping fixtures below the repository also works in minimal Pods.
        temp_parent = Path(__file__).resolve().parents[1] / ".test-tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        self.root = temp_parent / f"case-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.settings = Settings(
            repo_root=self.root,
            backend_root=self.root / "backend",
            frontend_root=self.root / "frontend",
            workspace_root=self.root,
            data_root=self.root / "data",
            output_root=self.root / "output",
            job_root=self.root / "jobs",
            config_root=self.root / "config",
            rclone_config=self.root / "rclone" / "rclone.conf",
            gdrive_remote="gdrive",
            rclone_bin="rclone",
            app_port=8001,
            sd_scripts_root=Path(__file__).resolve().parents[1] / "backend" / "sd-scripts",
            allow_external_dataset_paths=True,
            allow_mock_tagger=True,
        )
        self.dataset = self.settings.data_root / "my_character"
        (self.dataset / "nested").mkdir(parents=True)
        (self.dataset / "001.png").write_bytes(b"not-an-image-but-a-valid-fixture")
        (self.dataset / "nested" / "002.jpg").write_bytes(b"fixture")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_dataset_and_drive_path_validation(self) -> None:
        self.assertEqual(validate_local_dataset("data/my_character", self.settings), self.dataset.resolve())
        self.assertEqual(normalize_local_path("data/my_character", self.settings), self.dataset.resolve())
        self.assertEqual(
            validate_gdrive_path("gdrive:sdxl_lora/datasets/my_character", "gdrive"),
            "sdxl_lora/datasets/my_character",
        )
        self.assertEqual(
            gdrive_uri("sdxl_lora/datasets/my_character", self.settings),
            "gdrive:sdxl_lora/datasets/my_character",
        )
        with self.assertRaises(ValueError):
            validate_gdrive_path("sdxl_lora/../secrets", "gdrive")
        with self.assertRaises(ValueError):
            validate_gdrive_path("other:somewhere", "gdrive")

        counts = dataset_counts(self.dataset)
        self.assertEqual(counts, {"image_count": 2, "caption_count": 0, "missing_caption_count": 2})

    def test_dataset_resync_clears_only_the_target_directory(self) -> None:
        target = dataset_directory_for_name("resync_case", self.settings)
        stale = target / "old.txt"
        nested_stale = target / "old" / "nested.bin"
        nested_stale.parent.mkdir(parents=True)
        stale.write_text("stale", encoding="utf-8")
        nested_stale.write_bytes(b"stale")

        removed = clear_dataset_directory(target, self.settings)

        self.assertEqual(removed, 2)
        self.assertTrue(target.is_dir())
        self.assertFalse(stale.exists())
        self.assertFalse(nested_stale.exists())
        self.assertTrue(self.settings.data_root.is_dir())
        with self.assertRaises(ValueError):
            clear_dataset_directory(self.settings.data_root, self.settings)
        outside = self.root / "outside-dataset"
        outside.mkdir()
        with self.assertRaises(ValueError):
            clear_dataset_directory(outside, self.settings)

    def test_runpod_dataset_operations_are_limited_to_local_data_root(self) -> None:
        restricted = replace(self.settings, allow_external_dataset_paths=False)
        external = self.root / "workspace-dataset"
        external.mkdir()
        with self.assertRaises(ValueError):
            validate_local_dataset(str(external), restricted)
        external_image = external / "outside.png"
        external_image.write_bytes(b"fixture")
        with self.assertRaises(ValueError):
            validate_image_path(str(external_image), restricted)

    def test_rclone_commands_are_copy_only(self) -> None:
        service = RcloneService(self.settings)
        dataset_command = service.dataset_command("sdxl_lora/datasets/my_character", self.dataset)
        caption_command = service.captions_command(self.dataset, "sdxl_lora/datasets/my_character")
        output_command = service.output_command(self.root / "output", "sdxl_lora/outputs/my_character")

        for command in (dataset_command, caption_command, output_command):
            self.assertEqual(command[3], "copy")
            self.assertNotIn("sync", command)
        self.assertIn("--include", caption_command)
        self.assertIn("*.txt", caption_command)
        self.assertIn("training_summary.txt", output_command)

    def test_caption_edit_batch_operations_and_nested_images(self) -> None:
        first = self.dataset / "001.png"
        second = self.dataset / "nested" / "002.jpg"
        self.assertEqual(update_image_tags(str(first), ["1girl", "blue eyes", "blue eyes"], self.settings), ["1girl", "blue eyes"])
        batch_update_tags(str(self.dataset), ["my_trigger"], "prepend", self.settings)
        batch_update_tags(str(self.dataset), ["masterpiece"], "append", self.settings)
        self.assertEqual(read_tags(first), ["my_trigger", "1girl", "blue eyes", "masterpiece"])
        self.assertEqual(read_tags(second), ["my_trigger", "masterpiece"])

        removed = batch_remove_tags(str(self.dataset), ["masterpiece"], self.settings)
        self.assertEqual(removed["images_changed"], 2)
        records = list_image_records(str(self.dataset), self.settings)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["relative_path"], "nested/002.jpg")

    def test_loss_parser_and_training_summary(self) -> None:
        parser = LossParser()
        parser.on_line("epoch 1/3:  10%|#         | 1/10 [00:01<00:09, loss=0.1234, avr_loss=0.2345, lr=1e-4]")
        parser.on_line("epoch 2/3:  20%|##        | 2/10 [00:02<00:08, loss=0.1000, avr_loss=0.2000, lr=9e-5]")
        self.assertEqual(parser.epoch, 2)
        self.assertEqual(parser.step, 2)
        self.assertAlmostEqual(parser.loss or 0, 0.1)
        self.assertAlmostEqual(parser.average_loss or 0, 0.2)
        self.assertEqual(len(parser.steps), 2)
        self.assertEqual(len(parser.epochs), 2)

        summary_path = self.root / "output" / "job" / "training_summary.txt"
        write_training_summary(
            summary_path,
            {
                "job_id": "job-1",
                "job_name": "demo",
                "status": "failed",
                "environment": {"gpu": "test"},
                "dataset": {"image_count": 2},
                "caption": {"trigger_words": "my_character_v1"},
                "training_config": {"epochs": 3},
                "loss_history": parser.as_dict() | {"final_loss": parser.loss},
                "loss_diagnostics": {
                    "total_training_steps": 30,
                    "effective_images_per_epoch": 20,
                    "images_x_repeats": 20,
                    "lowest_epoch_loss": 0.2,
                    "lowest_loss_epoch": 2,
                    "final_loss": parser.loss,
                },
                "output_sync": {
                    "enabled": True,
                    "gdrive_output_path": "sdxl_lora/outputs/demo",
                    "status": "failed",
                    "error": "rclone unavailable",
                },
                "output_files": ["lora/demo-000001.safetensors"],
                "warnings": ["automatic output sync failed"],
                "errors": ["CUDA out of memory"],
            },
        )
        summary = summary_path.read_text(encoding="utf-8")
        for section in (
            "[JOB INFO]",
            "[ENVIRONMENT]",
            "[DATASET]",
            "[TRAINING CONFIG]",
            "[LOSS DIAGNOSTICS]",
            "[LOSS HISTORY]",
            "[OUTPUT FILES]",
            "[OUTPUT SYNC]",
            "[ERRORS]",
        ):
            self.assertIn(section, summary)
        self.assertIn("status=failed", summary)
        self.assertIn("epoch_1_avg_loss=0.2345", summary)
        self.assertIn("final_loss=0.1", summary)
        self.assertIn("total_training_steps=30", summary)
        self.assertIn("trigger_words=my_character_v1", summary)
        self.assertIn("status=failed", summary.split("[OUTPUT SYNC]", 1)[1])
        self.assertIn("enabled=true", summary.split("[OUTPUT SYNC]", 1)[1])
        self.assertIn("gdrive_output_path=sdxl_lora/outputs/demo", summary)
        self.assertIn("error=rclone unavailable", summary)
        self.assertIn("automatic output sync failed", summary)

    def test_training_command_contains_epoch_samples_loss_and_save_flags(self) -> None:
        config = TrainingConfig(
            path=str(self.dataset),
            model="stabilityai/stable-diffusion-xl-base-1.0",
            name="demo lora",
            epochs=3,
            repeats=2,
            batch_size=2,
            sample_prompt="fixed prompt",
            sample_prompts=["second prompt"],
            sample_negative_prompt="low quality",
            sample_seed=42,
            sample_sampler="euler_a",
            sample_steps=12,
            trigger_word="my_char",
        )
        prompt_path = self.root / "jobs" / "sample_prompts.json"
        command = build_training_command(
            config,
            self.settings.sd_scripts_root / "sdxl_train_network.py",
            self.root / "jobs" / "dataset_config.toml",
            self.root / "output" / "job" / "lora",
            prompt_path,
            self.root / "output" / "job" / "logs" / "tensorboard",
            42,
        )
        self.assertIn("--save_every_n_epochs=1", command)
        self.assertIn("--sample_at_first", command)
        self.assertIn("--sample_every_n_epochs=1", command)
        self.assertIn(f"--sample_prompts={prompt_path}", command)
        self.assertIn("--sample_sampler=euler_a", command)
        self.assertIn("--log_with=tensorboard", command)
        self.assertIn("--train_batch_size=2", command)
        self.assertIn("--max_train_epochs=3", command)
        self.assertNotIn("sync", command)

        prompts = build_sample_prompts(config)
        self.assertEqual([item["prompt"] for item in prompts], ["my_char, fixed prompt", "my_char, second prompt"])
        self.assertTrue(all(item["seed"] == 42 for item in prompts))

        dataset_config = build_dataset_toml(self.dataset, 2, 1024, 1024, 256, 2048)
        self.assertIn("num_repeats = 2", dataset_config)
        self.assertIn("caption_extension", dataset_config)

    def test_wd14_skip_overwrite_and_counts(self) -> None:
        fresh_image = self.dataset / "fresh.webp"
        fresh_image.write_bytes(b"fixture")
        existing_image = self.dataset / "nested" / "002.jpg"
        existing_image.with_suffix(".txt").write_text("manual_tag", encoding="utf-8")

        result = run_mock_tagger([fresh_image, existing_image], overwrite_existing_captions=False)
        self.assertEqual(result, {"tagged_count": 1, "skipped_count": 1, "failed_count": 0})
        self.assertEqual(existing_image.with_suffix(".txt").read_text(encoding="utf-8"), "manual_tag")
        self.assertEqual(read_tags(fresh_image)[-1], "mock_tag")

        result = run_mock_tagger([existing_image], overwrite_existing_captions=True)
        self.assertEqual(result, {"tagged_count": 1, "skipped_count": 0, "failed_count": 0})
        self.assertEqual(read_tags(existing_image)[-1], "mock_tag")

    def test_trigger_word_is_applied_once_to_every_sample_prompt(self) -> None:
        config = TrainingConfig(
            path=str(self.dataset),
            model="stabilityai/stable-diffusion-xl-base-1.0",
            trigger_word="my_char",
            sample_prompt="",
            sample_prompts=["my_char, 1girl, solo", "looking at viewer"],
        )
        prompts = build_sample_prompts(config)
        self.assertEqual(
            [item["prompt"] for item in prompts],
            ["my_char, 1girl, solo", "my_char, looking at viewer"],
        )

        without_trigger = TrainingConfig(
            path=str(self.dataset),
            model="stabilityai/stable-diffusion-xl-base-1.0",
            sample_prompt="1girl, solo",
        )
        self.assertEqual(build_sample_prompts(without_trigger)[0]["prompt"], "1girl, solo")

    def test_tensorboard_epoch_average_uses_epoch_step_without_duplicates(self) -> None:
        parser = LossParser()
        parser.on_line("epoch 1/2: 50%|#####     | 1/2 [00:01, loss=0.3, avr_loss=0.25, lr=1e-4]")
        parser.on_line("epoch 2/2: 100%|##########| 2/2 [00:02, loss=0.1, avr_loss=0.15, lr=9e-5]")
        parser.merge_structured_records(
            [
                {"tag": "loss/current", "step": 1, "value": 0.31},
                {"tag": "loss/average", "step": 1, "value": 0.26},
                {"tag": "loss/epoch_average", "step": 1, "value": 0.25},
                {"tag": "loss/current", "step": 2, "value": 0.11},
                {"tag": "loss/epoch_average", "step": 2, "value": 0.15},
            ]
        )
        self.assertEqual(list(parser.epochs), [1, 2])
        self.assertEqual(len(parser.steps), 2)
        self.assertAlmostEqual(parser.loss or 0, 0.11)
        self.assertAlmostEqual(parser.epochs[1]["average_loss"], 0.25)
        self.assertAlmostEqual(parser.epochs[2]["average_loss"], 0.15)

    def test_training_loss_diagnostics_are_computed_from_observed_and_configured_values(self) -> None:
        parser = LossParser()
        parser.on_line("epoch 1/2: 50%|#####     | 5/10 [00:01, loss=0.3, avr_loss=0.25, lr=1e-4]")
        parser.on_line("epoch 2/2: 100%|##########| 10/10 [00:02, loss=0.1, avr_loss=0.15, lr=9e-5]")
        diagnostics = TrainingService._loss_diagnostics(
            {"status": "completed"},
            TrainingConfig(
                path=str(self.dataset),
                model="stabilityai/stable-diffusion-xl-base-1.0",
                epochs=2,
                repeats=3,
                batch_size=2,
                gradient_accumulation_steps=2,
            ),
            {"image_count": 5},
            parser,
        )
        self.assertEqual(diagnostics["total_training_steps"], 10)
        self.assertEqual(diagnostics["images_x_repeats"], 15)
        self.assertEqual(diagnostics["effective_images_per_epoch"], 15)
        self.assertEqual(diagnostics["estimated_total_steps"], 8)
        self.assertAlmostEqual(diagnostics["lowest_epoch_loss"], 0.15)
        self.assertEqual(diagnostics["lowest_loss_epoch"], 2)
        self.assertAlmostEqual(diagnostics["final_loss"], 0.1)

    def test_failed_automatic_output_sync_keeps_training_completed_and_is_written_to_summary(self) -> None:
        class StubManager:
            def __init__(self) -> None:
                self.on_done = None

            async def start(self, **kwargs):
                self.on_done = kwargs["on_done"]
                return ProcessRun(
                    run_id=kwargs["run_id"],
                    kind=kwargs["kind"],
                    command=kwargs["command"],
                    log_path=kwargs["log_path"],
                )

            async def wait(self, run):
                return run

        test_root = self.root

        class FailingSync:
            async def start_output(self, request):
                sync_run = ProcessRun("output-sync", "sync", [], test_root / "jobs" / "sync.log")
                return {"status": "failed", "error": "remote unavailable"}, sync_run

        manager = StubManager()
        service = TrainingService(self.settings, manager, EventHub(), FailingSync())
        config = TrainingConfig(
            path=str(self.dataset),
            model="stabilityai/stable-diffusion-xl-base-1.0",
            name="summary_sync_demo",
            epochs=2,
            repeats=2,
            trigger_word="my_character_v1",
            auto_sync_output=True,
            gdrive_output_path="sdxl_lora/outputs/summary_sync_demo",
        )

        state, _ = asyncio.run(service.start(config))
        completed_run = ProcessRun("training-done", "training", [], self.root / "jobs" / "done.log")
        completed_run.status = "succeeded"
        completed_run.returncode = 0
        completed_run.ended_at = "2026-08-30T00:00:01+00:00"
        asyncio.run(manager.on_done(completed_run))

        final_state = service.get(state["job_id"])
        self.assertEqual(final_state["status"], "completed")
        self.assertEqual(final_state["output_sync"]["status"], "failed")
        summary = Path(final_state["summary_path"]).read_text(encoding="utf-8")
        self.assertIn("status=completed", summary)
        self.assertIn("[OUTPUT SYNC]", summary)
        self.assertIn("status=failed", summary.split("[OUTPUT SYNC]", 1)[1])
        self.assertIn("remote unavailable", summary)
        self.assertIn("automatic output sync failed", summary)

    def test_summary_samples_large_step_history_and_keeps_boundaries(self) -> None:
        steps = [
            {
                "step": index + 1,
                "epoch": index // 100 + 1,
                "loss": 1.0 / (index + 1),
                "average_loss": None,
                "learning_rate": 1e-4,
            }
            for index in range(1200)
        ]
        sampled = representative_step_records(steps, 500)
        self.assertLessEqual(len(sampled), 500)
        self.assertEqual(sampled[0]["step"], 1)
        self.assertEqual(sampled[-1]["step"], 1200)
        self.assertIn(100, [item["step"] for item in sampled])
        self.assertIn(101, [item["step"] for item in sampled])

        summary_path = self.root / "output" / "job" / "training_summary.txt"
        write_training_summary(
            summary_path,
            {"loss_history": {"steps": steps, "epochs": [], "final_loss": steps[-1]["loss"]}, "step_loss_summary_limit": 500},
        )
        lines = summary_path.read_text(encoding="utf-8").splitlines()
        step_lines = [line for line in lines if line.startswith("step=")]
        self.assertEqual(len(step_lines), len(sampled))
        self.assertIn("step_records_total=1200", lines)

    def test_recommendation_includes_labeled_estimated_steps(self) -> None:
        recommendation = recommend_settings("20", "character", "balanced", 2)
        self.assertEqual(recommendation["estimated_total_steps_label"], "estimated")
        self.assertEqual(recommendation["estimated_total_steps"], 1500)


if __name__ == "__main__":
    unittest.main()
