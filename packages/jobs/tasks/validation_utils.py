import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class SOTDataValidator:
    
    def __init__(self, download_path: Path):
        self.download_path = download_path
        self.meta_path = download_path / "META.json"
    
    def validate_meta_exists(self) -> bool:
        if not self.meta_path.exists():
            logger.error("META.json not found", path=str(self.meta_path))
            return False
        logger.info("META.json found", path=str(self.meta_path))
        return True
    
    def load_meta(self) -> Optional[Dict]:
        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            logger.info("META.json loaded", files_count=len(meta.get('files', [])))
            return meta
        except Exception as e:
            logger.error("Failed to load META.json", error=str(e))
            return None
    
    def validate_checksums(self, meta: Dict) -> bool:
        files = meta.get('files', [])
        
        for file_info in files:
            filename = file_info['filename']
            expected_checksum = file_info.get('checksum')
            
            if not expected_checksum:
                logger.warning("No checksum for file", filename=filename)
                continue
            
            file_path = self.download_path / filename
            if not file_path.exists():
                logger.error("File missing", filename=filename)
                return False
            
            actual_checksum = self._calculate_checksum(file_path)
            if actual_checksum != expected_checksum:
                logger.error(
                    "Checksum mismatch",
                    filename=filename,
                    expected=expected_checksum,
                    actual=actual_checksum
                )
                return False
            
            logger.info("Checksum validated", filename=filename)
        
        return True
    
    def validate_all_files_present(self, meta: Dict) -> bool:
        files = meta.get('files', [])
        missing_files = []
        
        for file_info in files:
            filename = file_info['filename']
            file_path = self.download_path / filename
            
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.error(
                "Missing files detected",
                count=len(missing_files),
                files=missing_files
            )
            return False
        
        logger.info("All files present", count=len(files))
        return True
    
    def _calculate_checksum(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def cleanup_download_dir(self):
        if self.download_path.exists():
            import shutil
            shutil.rmtree(self.download_path)
            logger.info("Cleaned up download directory", path=str(self.download_path))