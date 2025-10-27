"""Parse Cell Painting Gallery load_data.csv files and group by plate."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)


class LoadDataParser:
    """Parser for Cell Painting Gallery load_data.csv files."""

    # Default channel names and columns
    CHANNELS = ["DNA", "RNA", "ER", "AGP", "Mito"]
    URL_COLUMNS = {
        "DNA": "URL_OrigDNA",
        "RNA": "URL_OrigRNA",
        "ER": "URL_OrigER",
        "AGP": "URL_OrigAGP",
        "Mito": "URL_OrigMito",
    }
    METADATA_COLUMNS = [
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
    ]

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        url_columns: Optional[Dict[str, str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """
        Initialize the load_data.csv parser.

        Args:
            channels: List of channel names (default: DNA, RNA, ER, AGP, Mito)
            url_columns: Mapping of channel names to URL column names
            metadata_columns: List of metadata column names to extract
        """
        self.channels = channels or self.CHANNELS
        self.url_columns = url_columns or self.URL_COLUMNS
        self.metadata_columns = metadata_columns or self.METADATA_COLUMNS

    def parse(self, csv_path: str) -> pd.DataFrame:
        """
        Parse load_data.csv file.

        Args:
            csv_path: Path to load_data.csv (local path or S3 URI)

        Returns:
            DataFrame with parsed data

        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If required columns are missing
        """
        logger.info(f"Parsing load_data.csv from: {csv_path}")

        # Read CSV (pandas handles S3 URIs with s3fs installed)
        try:
            # Configure s3fs for anonymous access if S3 path
            if csv_path.startswith("s3://"):
                import s3fs
                fs = s3fs.S3FileSystem(anon=True)
                with fs.open(csv_path, 'rb') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(csv_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read {csv_path}: {e}")

        # Validate required columns exist
        required_cols = self.metadata_columns + list(self.url_columns.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Loaded {len(df)} rows from load_data.csv")
        return df

    def group_by_plate(
        self, df: pd.DataFrame
    ) -> Dict[str, List[Dict]]:
        """
        Group sites by plate.

        Args:
            df: DataFrame from parse()

        Returns:
            Dictionary mapping plate_id to list of site information:
            {
                "BR00117036": [
                    {
                        "source": "source_1",
                        "batch": "20200101_batch1",
                        "plate": "BR00117036",
                        "well": "A01",
                        "site": "1",
                        "channels": {
                            "DNA": "s3://cellpainting-gallery/.../DNA.tif",
                            "RNA": "s3://cellpainting-gallery/.../RNA.tif",
                            ...
                        }
                    },
                    ...
                ]
            }
        """
        logger.info("Grouping sites by plate...")

        grouped = {}
        for _, row in df.iterrows():
            plate_id = row["Metadata_Plate"]

            # Extract site metadata
            site_info = {
                "source": row["Metadata_Source"],
                "batch": row["Metadata_Batch"],
                "plate": plate_id,
                "well": row["Metadata_Well"],
                "site": str(row["Metadata_Site"]),
            }

            # Extract channel URLs
            channels = {}
            for channel_name in self.channels:
                url_col = self.url_columns[channel_name]
                url = row[url_col]

                # Handle both S3 and local paths
                if pd.notna(url):
                    channels[channel_name] = str(url)
                else:
                    logger.warning(
                        f"Missing URL for {channel_name} in plate={plate_id}, "
                        f"well={site_info['well']}, site={site_info['site']}"
                    )

            site_info["channels"] = channels

            # Group by plate
            if plate_id not in grouped:
                grouped[plate_id] = []
            grouped[plate_id].append(site_info)

        logger.info(f"Grouped {len(df)} sites into {len(grouped)} plates")
        for plate_id, sites in list(grouped.items())[:3]:  # Show first 3 plates
            logger.info(f"  Plate {plate_id}: {len(sites)} sites")

        return grouped

    def extract_plate_s3_prefix(self, plate_sites: List[Dict]) -> Optional[str]:
        """
        Extract the S3 prefix for downloading an entire plate.

        Args:
            plate_sites: List of site info dicts for a single plate

        Returns:
            S3 prefix path (e.g., "cpg0016-jump/source_1/images/batch1/plate1/")
            or None if not an S3 path
        """
        if not plate_sites:
            return None

        # Get first channel URL from first site
        first_site = plate_sites[0]
        first_channel = list(first_site["channels"].values())[0]

        # Check if it's an S3 URL
        parsed = urlparse(first_channel)
        if parsed.scheme != "s3":
            return None

        # Extract bucket and prefix
        # Example URL: s3://cellpainting-gallery/cpg0016-jump/source_1/images/batch/plate/file.tif
        # We want: cpg0016-jump/source_1/images/batch/plate/
        path_parts = Path(parsed.path.lstrip("/")).parts

        # Find the plate directory (typically the parent of the file)
        # Structure is usually: .../images/<batch>/<plate>/<file.tif>
        if len(path_parts) >= 2:
            # Remove the filename, keep everything up to plate directory
            plate_prefix = "/".join(path_parts[:-1]) + "/"
            return plate_prefix

        return None

    def get_plate_summary(self, grouped: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Get summary statistics for each plate.

        Args:
            grouped: Output from group_by_plate()

        Returns:
            DataFrame with plate-level summary statistics
        """
        summary_data = []
        for plate_id, sites in grouped.items():
            summary_data.append(
                {
                    "plate_id": plate_id,
                    "num_sites": len(sites),
                    "source": sites[0]["source"] if sites else None,
                    "batch": sites[0]["batch"] if sites else None,
                    "num_wells": len(set(s["well"] for s in sites)),
                }
            )

        return pd.DataFrame(summary_data)


def parse_load_data_csv(
    csv_path: str,
    channels: Optional[List[str]] = None,
    url_columns: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Convenience function to parse load_data.csv and group by plate.

    Args:
        csv_path: Path to load_data.csv (local or S3)
        channels: Optional list of channel names
        url_columns: Optional mapping of channel names to URL columns

    Returns:
        Dictionary mapping plate_id to list of site information
    """
    parser = LoadDataParser(channels=channels, url_columns=url_columns)
    df = parser.parse(csv_path)
    return parser.group_by_plate(df)
