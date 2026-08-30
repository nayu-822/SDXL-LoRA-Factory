from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from backend.app.config import Settings
from backend.app.loss_parser import LossParser
from backend.app.models import TrainingConfig
from backend.app.paths import (
    dataset_counts,
    gdrive_uri,
    normalize_local_path,
    validate_gdrive_path,
    validate_local_dataset,
)
from backend.app.rclone_sync import RcloneService
from backend.app.summary import write_training_summary
from backend.app.tag_ops import (
    batch_remove_tags,
    batch_update_tags,
    list_image_records,
    read_tags,
    update_image_tags,
)
from backend.app.training import build_dataset_toml, build_sample_prompts, build_training_command
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
                "training_config": {"epochs": 3},
                "loss_history": parser.as_dict() | {"final_loss": parser.loss},
                "output_files": ["lora/demo-000001.safetensors"],
                "errors": ["CUDA out of memory"],
            },
        )
        summary = summary_path.read_text(encoding="utf-8")
        for section in ("[JOB INFO]", "[ENVIRONMENT]", "[DATASET]", "[TRAINING CONFIG]", "[LOSS HISTORY]", "[OUTPUT FILES]", "[ERRORS]"):
            self.assertIn(section, summary)
        self.assertIn("status=failed", summary)
        self.assertIn("epoch_1_avg_loss=0.2345", summary)
        self.assertIn("final_loss=0.1", summary)

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
        self.assertEqual([item["prompt"] for item in prompts], ["fixed prompt", "second prompt"])
        self.assertTrue(all(item["seed"] == 42 for item in prompts))

        dataset_config = build_dataset_toml(self.dataset, 2, 1024, 1024, 256, 2048)
        self.assertIn("num_repeats = 2", dataset_config)
        self.assertIn("caption_extension", dataset_config)

    def test_explicit_mock_tagger_writes_captions(self) -> None:
        nested_image = self.dataset / "nested" / "002.jpg"
        self.assertEqual(run_mock_tagger([nested_image]), 0)
        self.assertEqual(read_tags(nested_image)[-1], "mock_tag")


if __name__ == "__main__":
    unittest.main()
