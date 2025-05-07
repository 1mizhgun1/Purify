import subprocess
import os
import soundfile as sf
import torch
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
from typing import Optional, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "lorenzoncina/whisper-small-ru"
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.processor.feature_extractor.return_attention_mask = True
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def download_audio(
        self,
        url: str,
        output_file: str = "/app/output/downloaded_audio.wav",
        max_duration: int = 60,
        retries: int = 3,
        quiet: bool = True
    ) -> bool:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        for attempt in range(retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{retries} for {url}")
                
                ffmpeg_cmd = [
                    "yt-dlp",
                    "-x", "--audio-format", "wav",
                    "--audio-quality", "0",
                    "--external-downloader", "ffmpeg",
                    "--external-downloader-args", f"ffmpeg_i:-ss 0 -t {max_duration}",
                    "-o", output_file,
                    url
                ]
                
                subprocess_args = {
                    'stdout': subprocess.PIPE if quiet else None,
                    'stderr': subprocess.PIPE if quiet else None
                }
                
                result = subprocess.run(ffmpeg_cmd, **subprocess_args)
                
                if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    logger.info("Audio downloaded successfully")
                    return True
                
                logger.warning("Primary download method failed, trying fallback...")
                temp_file = f"{output_file}.temp"
                subprocess.run([
                    "yt-dlp", "-x", "--audio-format", "wav", "-o", temp_file, url
                ], **subprocess_args)
                
                subprocess.run([
                    "ffmpeg", "-i", f"{temp_file}.wav", "-ss", "0", "-t", str(max_duration),
                    "-acodec", "copy", "-y", output_file
                ], **subprocess_args)
                
                if os.path.exists(f"{temp_file}.wav"):
                    os.remove(f"{temp_file}.wav")
                
                if os.path.exists(output_file):
                    logger.info("Audio downloaded using fallback method")
                    return True
                    
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
                if os.path.exists(output_file):
                    os.remove(output_file)
        
        logger.error("All download attempts failed")
        return False

    def _preprocess_audio(self, audio_path: str, sampling_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """Вспомогательный метод для предварительной обработки аудио"""
        try:
            audio, file_sr = sf.read(audio_path)
            
            # Проверка наличия аудиоданных
            if len(audio) == 0:
                raise ValueError("Audio file is empty")
            
            # Конвертация стерео в моно
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Ресемплинг при необходимости
            if file_sr != sampling_rate:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sampling_rate)
            
            # Нормализация аудио
            audio = librosa.util.normalize(audio)
            
            # Проверка минимальной длины
            min_samples = sampling_rate // 2  # Минимум 0.5 секунды
            if len(audio) < min_samples:
                logger.warning(f"Audio too short ({len(audio)} samples), padding to {min_samples}")
                audio = np.pad(audio, (0, max(0, min_samples - len(audio))), mode='constant')
            
            return audio, sampling_rate
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise ValueError(f"Audio preprocessing error: {str(e)}")
    
    def _replace_punctuation(self, text: str) -> str:
        """Заменяет все знаки препинания на пробелы в тексте"""
        # Расширенное регулярное выражение для пунктуации
        punctuation_pattern = r'[\.,\/#!%\^&\*;:{}=\-_`~()\[\]"\'<>?\|«»„“¨‘’…—–¬\+\$\€£¥©®™•]'
        text = re.sub(punctuation_pattern, ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def transcribe_audio(self, audio_path: str, sampling_rate: int = 16000, max_chunks: int = 10, chunk_duration: int = 30) -> str:
        try:
            logger.info(f"Starting transcription for {audio_path}")
            
            audio, sr = self._preprocess_audio(audio_path, sampling_rate)
            logger.info(f"Audio loaded: {len(audio)} samples ({len(audio)/sr:.2f}s)")
            
            samples_per_chunk = chunk_duration * sampling_rate
            total_samples = len(audio)
            
            # Обработка для одного чанка
            if total_samples <= samples_per_chunk:
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=sr,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=samples_per_chunk,
                        truncation=True,
                        return_attention_mask=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            input_features=inputs.input_features,
                            attention_mask=inputs.attention_mask,
                            forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                                language="russian",
                                task="transcribe"
                            )
                        )
                    
                    transcription = self.processor.decode(output_ids[0], skip_special_tokens=True)
                    transcription = self._replace_punctuation(transcription)  # Очистка здесь
                    logger.info("Transcription completed successfully")
                    return transcription
                
                except Exception as e:
                    logger.error(f"Transcription failed for single chunk: {str(e)}")
                    raise ValueError(f"Transcription error: {str(e)}")
            
            # Обработка нескольких чанков
            transcriptions = []
            total_chunks = min(max_chunks, (total_samples + samples_per_chunk - 1) // samples_per_chunk)
            logger.info(f"Processing {total_chunks} chunks")
            
            for chunk_idx in range(total_chunks):
                try:
                    start = chunk_idx * samples_per_chunk
                    end = min(start + samples_per_chunk, total_samples)
                    chunk = audio[start:end]
                    
                    inputs = self.processor(
                        chunk,
                        sampling_rate=sr,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=samples_per_chunk,
                        truncation=True,
                        return_attention_mask=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            input_features=inputs.input_features,
                            attention_mask=inputs.attention_mask,
                            forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                                language="russian",
                                task="transcribe"
                            )
                        )
                    
                    chunk_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
                    chunk_text = self._replace_punctuation(chunk_text)  # Очистка каждого чанка
                    transcriptions.append(chunk_text)
                    logger.info(f"Chunk {chunk_idx + 1}/{total_chunks} processed")
                
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_idx + 1}: {str(e)}")
                    transcriptions.append(f"[CHUNK_{chunk_idx + 1}_ERROR]")

            # Дополнительная очистка финального текста
            full_text = " ".join(transcriptions)
            clean_text = self._replace_punctuation(full_text)
            
            # Логирование для отладки
            logger.debug(f"Original text: {full_text}")
            logger.debug(f"Cleaned text: {clean_text}")
            
            return clean_text
        
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise ValueError(f"Transcription error: {str(e)}")