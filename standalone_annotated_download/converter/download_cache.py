"""Download cache manager for tracking downloaded plates."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DownloadCache:
    """
    Manage download cache to avoid re-downloading plates.

    Each plate gets a directory in the cache with a .download_complete marker
    file to indicate successful download. A central download_log.json tracks
    all downloaded plates with timestamps.
    """

    DOWNLOAD_COMPLETE_MARKER = ".download_complete"
    DOWNLOAD_LOG_FILE = "download_log.json"

    def __init__(self, cache_dir: Path):
        """
        Initialize download cache manager.

        Args:
            cache_dir: Root directory for caching downloaded plates
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.cache_dir / self.DOWNLOAD_LOG_FILE
        self.log = self._load_log()

    def _load_log(self) -> Dict:
        """Load download log from disk."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load download log: {e}, creating new log")
                return {"plates": {}, "created": datetime.now().isoformat()}
        else:
            return {"plates": {}, "created": datetime.now().isoformat()}

    def _save_log(self):
        """Save download log to disk."""
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save download log: {e}")

    def get_plate_cache_dir(self, plate_id: str) -> Path:
        """
        Get cache directory for a specific plate.

        Args:
            plate_id: Plate identifier

        Returns:
            Path to plate cache directory
        """
        plate_dir = self.cache_dir / plate_id
        return plate_dir

    def is_plate_downloaded(self, plate_id: str) -> bool:
        """
        Check if a plate has been fully downloaded.

        Args:
            plate_id: Plate identifier

        Returns:
            True if plate is fully downloaded, False otherwise
        """
        plate_dir = self.get_plate_cache_dir(plate_id)
        marker_file = plate_dir / self.DOWNLOAD_COMPLETE_MARKER

        # Check for marker file
        if marker_file.exists():
            logger.debug(f"Plate {plate_id} already downloaded (marker exists)")
            return True

        # Also check download log
        if plate_id in self.log.get("plates", {}):
            # Verify marker file exists (in case log is out of sync)
            if not marker_file.exists():
                logger.warning(
                    f"Plate {plate_id} in log but marker missing, "
                    f"will re-download"
                )
                # Remove from log
                del self.log["plates"][plate_id]
                self._save_log()
                return False
            return True

        return False

    def mark_plate_complete(
        self,
        plate_id: str,
        num_files: Optional[int] = None,
        size_bytes: Optional[int] = None,
        s3_prefix: Optional[str] = None,
    ):
        """
        Mark a plate as fully downloaded.

        Args:
            plate_id: Plate identifier
            num_files: Number of files downloaded (optional)
            size_bytes: Total size of downloaded files in bytes (optional)
            s3_prefix: S3 prefix where files were downloaded from (optional)
        """
        plate_dir = self.get_plate_cache_dir(plate_id)
        plate_dir.mkdir(parents=True, exist_ok=True)

        # Create marker file
        marker_file = plate_dir / self.DOWNLOAD_COMPLETE_MARKER
        with open(marker_file, "w") as f:
            marker_data = {
                "plate_id": plate_id,
                "completed_at": datetime.now().isoformat(),
                "num_files": num_files,
                "size_bytes": size_bytes,
                "s3_prefix": s3_prefix,
            }
            json.dump(marker_data, f, indent=2)

        # Update download log
        self.log.setdefault("plates", {})[plate_id] = {
            "completed_at": datetime.now().isoformat(),
            "num_files": num_files,
            "size_bytes": size_bytes,
            "s3_prefix": s3_prefix,
        }
        self.log["last_updated"] = datetime.now().isoformat()
        self._save_log()

        logger.info(f"Marked plate {plate_id} as downloaded")

    def get_downloaded_plates(self) -> Set[str]:
        """
        Get set of all downloaded plate IDs.

        Returns:
            Set of plate IDs that have been downloaded
        """
        return set(self.log.get("plates", {}).keys())

    def get_plate_info(self, plate_id: str) -> Optional[Dict]:
        """
        Get information about a downloaded plate.

        Args:
            plate_id: Plate identifier

        Returns:
            Dictionary with plate download info, or None if not downloaded
        """
        return self.log.get("plates", {}).get(plate_id)

    def remove_plate(self, plate_id: str, delete_files: bool = False):
        """
        Remove a plate from the download cache.

        Args:
            plate_id: Plate identifier
            delete_files: If True, also delete the cached files
        """
        # Remove from log
        if plate_id in self.log.get("plates", {}):
            del self.log["plates"][plate_id]
            self._save_log()

        # Remove marker file
        plate_dir = self.get_plate_cache_dir(plate_id)
        marker_file = plate_dir / self.DOWNLOAD_COMPLETE_MARKER
        if marker_file.exists():
            marker_file.unlink()

        # Optionally delete all files
        if delete_files and plate_dir.exists():
            import shutil

            shutil.rmtree(plate_dir)
            logger.info(f"Deleted cache directory for plate {plate_id}")
        else:
            logger.info(f"Removed plate {plate_id} from download log")

    def get_summary(self) -> Dict:
        """
        Get summary statistics about the download cache.

        Returns:
            Dictionary with cache statistics
        """
        plates = self.log.get("plates", {})
        total_size = sum(
            p.get("size_bytes", 0) for p in plates.values() if p.get("size_bytes")
        )
        total_files = sum(
            p.get("num_files", 0) for p in plates.values() if p.get("num_files")
        )

        return {
            "num_plates": len(plates),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3) if total_size else 0,
            "total_files": total_files,
            "cache_dir": str(self.cache_dir),
            "created": self.log.get("created"),
            "last_updated": self.log.get("last_updated"),
        }

    def list_plates(self) -> List[str]:
        """
        List all downloaded plate IDs.

        Returns:
            List of plate IDs
        """
        return sorted(self.get_downloaded_plates())

    def cleanup_incomplete_downloads(self):
        """
        Clean up incomplete downloads (directories without markers).

        This removes directories in the cache that don't have a
        .download_complete marker file.
        """
        if not self.cache_dir.exists():
            return

        cleaned = 0
        for plate_dir in self.cache_dir.iterdir():
            if not plate_dir.is_dir():
                continue

            # Skip if it's not a plate directory
            if plate_dir.name == "." or plate_dir.name.startswith("."):
                continue

            # Check for marker
            marker_file = plate_dir / self.DOWNLOAD_COMPLETE_MARKER
            if not marker_file.exists():
                logger.info(f"Removing incomplete download: {plate_dir.name}")
                import shutil

                shutil.rmtree(plate_dir)
                cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} incomplete download(s)")
