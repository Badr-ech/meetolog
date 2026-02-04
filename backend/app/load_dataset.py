from datasets import load_dataset
from pathlib import Path
import sys


def extract_audio_files(
	dataset_name: str,
	split: str = "train",
	dest_dir: str = "../dataset_audio",
	max_files: int | None = 100,
):
	"""Extract audio files from HuggingFace dataset.
	
	This approach reads the raw audio bytes from the dataset and saves them
	as files, avoiding the need for torchcodec audio decoding.
	"""
	print(f"Loading dataset {dataset_name} split={split} (streaming mode)...")
	
	dest_path = Path(__file__).parent.joinpath(dest_dir).resolve()
	dest_path.mkdir(parents=True, exist_ok=True)

	copied = 0
	print(f"Extracting audio files into: {dest_path}")
	
	try:
		# Load in streaming mode and access raw audio bytes
		ds = load_dataset(dataset_name, split=split, streaming=True)
		
		for i, example in enumerate(ds):
			if max_files is not None and copied >= max_files:
				break
			
			try:
				# Try to get audio data - it should have 'bytes' or 'path' field
				audio_data = example.get("audio")
				
				if audio_data is None:
					print(f"[{i}] No audio field found, skipping")
					continue
				
				# Check if it's a dict with 'bytes' or 'path'
				if isinstance(audio_data, dict):
					# If there's a 'bytes' field, save it directly
					if "bytes" in audio_data and audio_data["bytes"]:
						audio_bytes = audio_data["bytes"]
						# Determine extension from path if available
						ext = ".wav"  # default
						if "path" in audio_data and audio_data["path"]:
							ext = Path(audio_data["path"]).suffix or ".wav"
						
						dest_file = dest_path.joinpath(f"{copied:06d}_audio{ext}")
						dest_file.write_bytes(audio_bytes)
						copied += 1
						
						if copied % 10 == 0:
							print(f"Extracted {copied} files...")
						continue
					
					# If there's a 'path' field pointing to a local file
					if "path" in audio_data and audio_data["path"]:
						src_path = Path(audio_data["path"])
						if src_path.exists():
							import shutil
							dest_file = dest_path.joinpath(f"{copied:06d}_{src_path.name}")
							shutil.copy2(src_path, dest_file)
							copied += 1
							
							if copied % 10 == 0:
								print(f"Copied {copied} files...")
							continue
				
				print(f"[{i}] Could not extract audio data (no bytes or path field)")
				
			except Exception as e:
				print(f"[{i}] Error processing example: {e}")
				continue
	
	except Exception as e:
		print(f"Error loading dataset: {e}")
		print("\nNote: This dataset requires torchcodec for audio decoding.")
		print("Install FFmpeg full-shared build to enable torchcodec support.")
		return

	print(f"Done. Copied {copied} files to {dest_path}")


if __name__ == "__main__":
	# Usage: python load_dataset.py [max_files]
	max_files = None
	if len(sys.argv) > 1:
		try:
			max_files = int(sys.argv[1])
		except ValueError:
			print("Invalid max_files argument, must be integer")
			sys.exit(1)

	extract_audio_files(
		"hhoangphuoc/ami-av",
		split="train",
		dest_dir="../dataset_audio",
		max_files=max_files if max_files is not None else 100,
	)
