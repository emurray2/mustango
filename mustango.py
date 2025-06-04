import os
import json
import torch
import numpy as np
from huggingface_hub import snapshot_download
from tensorizer import TensorDeserializer, TensorSerializer

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from transformers import AutoTokenizer, T5ForConditionalGeneration
from modelling_deberta_v2 import DebertaV2ForTokenClassificationRegression

from diffusers import DDPMScheduler
from models import MusicAudioDiffusion


class MusicFeaturePredictor:
    def __init__(self, path, device="cuda:0", cache_dir=None, local_files_only=False):
        self.beats_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.beats_model = DebertaV2ForTokenClassificationRegression.from_pretrained(
            "microsoft/deberta-v3-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.beats_model.eval()
        self.beats_model.to(device)

        beats_ckpt = f"{path}/beats/microsoft-deberta-v3-large.pt"
        beats_weight = torch.load(beats_ckpt, map_location="cpu")
        self.beats_model.load_state_dict(beats_weight)

        self.chords_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.chords_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.chords_model.eval()
        self.chords_model.to(device)

        chords_ckpt = f"{path}/chords/flan-t5-large.bin"
        chords_weight = torch.load(chords_ckpt, map_location="cpu")
        self.chords_model.load_state_dict(chords_weight)

    def generate_beats(self, prompt):
        tokenized = self.beats_tokenizer(
            prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized = {k: v.to(self.beats_model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            out = self.beats_model(**tokenized)

        max_beat = (
            1 + torch.argmax(out["logits"][:, 0, :], -1).detach().cpu().numpy()
        ).tolist()[0]
        intervals = (
            out["values"][:, :, 0]
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
            .round(4)
            .tolist()
        )

        intervals = np.cumsum(intervals)
        predicted_beats_times = []
        for t in intervals:
            if t < 10:
                predicted_beats_times.append(round(t, 2))
            else:
                break
        predicted_beats_times = list(np.array(predicted_beats_times)[:50])

        if len(predicted_beats_times) == 0:
            predicted_beats = [[], []]
        else:
            beat_counts = []
            for i in range(len(predicted_beats_times)):
                beat_counts.append(float(1.0 + np.mod(i, max_beat)))
            predicted_beats = [[predicted_beats_times, beat_counts]]

        return max_beat, predicted_beats_times, predicted_beats

    def generate(self, prompt):
        max_beat, predicted_beats_times, predicted_beats = self.generate_beats(prompt)

        chords_prompt = "Caption: {} \\n Timestamps: {} \\n Max Beat: {}".format(
            prompt,
            " , ".join([str(round(t, 2)) for t in predicted_beats_times]),
            max_beat,
        )

        tokenized = self.chords_tokenizer(
            chords_prompt,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.chords_model.device) for k, v in tokenized.items()}

        generated_chords = self.chords_model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            min_length=8,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=1,
        )

        generated_chords = self.chords_tokenizer.decode(
            generated_chords[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).split(" n ")

        predicted_chords, predicted_chords_times = [], []
        for item in generated_chords:
            c, ct = item.split(" at ")
            predicted_chords.append(c)
            predicted_chords_times.append(float(ct))

        return predicted_beats, predicted_chords, predicted_chords_times

class Mustango:
    def __init__(
        self,
        name="declare-lab/mustango",
        cache_dir="./mustango_model",
        device="cuda:0",
        first_time_download=True,
    ):
        self.device = device
        self.first_time_download = first_time_download

        if first_time_download:
            # --- Download model ---
            print(f"Downloading {name} into {cache_dir} ...")
            path = snapshot_download(
                repo_id=name,
                cache_dir=cache_dir,
                local_files_only=False,
            )
            # --- Create music predictor ---
            self.music_model = MusicFeaturePredictor(
                path, device, cache_dir=cache_dir, local_files_only=False
            )
            # --- Load configs ---
            vae_config = json.load(open(f"{path}/configs/vae_config.json"))
            stft_config = json.load(open(f"{path}/configs/stft_config.json"))
            main_config = json.load(open(f"{path}/configs/main_config.json"))
            # --- Instantiate models ---
            self.vae = AutoencoderKL(**vae_config).to(device)
            self.stft = TacotronSTFT(**stft_config).to(device)
            self.model = MusicAudioDiffusion(
                main_config["text_encoder_name"],
                main_config["scheduler_name"],
                unet_model_config_path=os.path.join(self.path, "configs/music_diffusion_model_config.json"),
            ).to(device)
            # --- Get the weights ---
            vae_weights = torch.load(
            f"{path}/vae/pytorch_model_vae.bin", map_location=device
            )
            stft_weights = torch.load(
                f"{path}/stft/pytorch_model_stft.bin", map_location=device
            )
            main_weights = torch.load(
                f"{path}/ldm/pytorch_model_ldm.bin", map_location=device
            )
            # --- Load weights into models ---
            self.vae.load_state_dict(vae_weights)
            self.stft.load_state_dict(stft_weights)
            self.model.load_state_dict(main_weights)
            # --- Serialize the models (for faster loading later) ---
            self._save_component(self.vae, path, "vae", "pytorch_model_vae", "VAE")
            self._save_component(self.stft, path, "stft", "pytorch_model_stft", "STFT")
            self._save_component(self.model, path, "ldm", "pytorch_model_ldm", "LDM")
        else:
            # --- Get path without downloading ---
            print(f"Loading {name} using tensorizer ...")
            path = snapshot_download(
                repo_id=name,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            # --- Create music predictor ---
            self.music_model = MusicFeaturePredictor(
                path, device, cache_dir=cache_dir, local_files_only=True
            )
             # --- Deserialize weights (for faster loading) ---
            self._load_component(self.vae, path, "vae", "pytorch_model_vae", "VAE")
            self._load_component(self.stft, path, "stft", "pytorch_model_stft", "STFT")
            self._load_component(self.model, path, "ldm", "pytorch_model_ldm", "LDM")

        print("Successfully loaded checkpoint from:", name)

        self.vae.eval()
        self.stft.eval()
        self.model.eval()

        self.scheduler = DDPMScheduler.from_pretrained(
            main_config["scheduler_name"], subfolder="scheduler"
        )

    def _load_component(self, model, path, subdir, base_name):
        folder = os.path.join(self.path, subdir)
        tensor_path = os.path.join(folder, f"{base_name}.tensors")
        print(f"🔁 Loading tensorized {base_name} from {tensor_path}")
        deserializer = TensorDeserializer(tensor_path)
        deserializer.load_into_module(model)

    def _save_component(self, model, path, subdir, base_name):
        folder = os.path.join(self.path, subdir)
        os.makedirs(folder, exist_ok=True)
        tensor_path = os.path.join(folder, f"{base_name}.tensors")
        print(f"💾 Saving tensorized {base_name} to {tensor_path}")
        serializer = TensorSerializer(tensor_path)
        serializer.write_module(model)
        serializer.close()

    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        with torch.no_grad():
            beats, chords, chords_times = self.music_model.generate(prompt)
            latents = self.model.inference(
                [prompt],
                beats,
                [chords],
                [chords_times],
                self.scheduler,
                steps,
                guidance,
                samples,
                disable_progress,
            )
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0]
