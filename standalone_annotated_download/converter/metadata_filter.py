"""
Metadata filter for selective downloading based on DuckDB queries.

This module provides a high-level interface to query the JUMP metadata
database and generate download manifests for filtered plate subsets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Filter JUMP metadata to generate selective download manifests.

    Features:
    - Filter by plate type (COMPOUND, ORF, CRISPR, etc.)
    - Filter by perturbation modality
    - Filter by specific compounds (JCP2022 IDs)
    - Filter by SMILES patterns (requires RDKit)
    - Balanced sampling across categories
    - Generate download manifests with S3 paths

    Example:
        >>> filter = MetadataFilter(db_path)
        >>> plates = filter.filter_by_plate_type(['COMPOUND'])
        >>> print(f"Found {len(plates)} COMPOUND plates")
    """

    # S3 base paths
    S3_BUCKET = "cellpainting-gallery"
    S3_PROJECT = "cpg0016-jump"

    def __init__(self, db_path: str):
        """
        Initialize metadata filter.

        Args:
            db_path: Path to DuckDB metadata database

        Raises:
            FileNotFoundError: If database doesn't exist
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.db_path = db_path
        self.con = duckdb.connect(str(db_path), read_only=True)

        logger.info(f"Connected to metadata database: {db_path}")

        # Cache table info
        self._cache_table_info()

    def _cache_table_info(self):
        """Cache basic table information for quick reference."""
        self.n_plates = self.con.execute("SELECT COUNT(*) FROM plate").fetchone()[0]
        self.n_wells = self.con.execute("SELECT COUNT(*) FROM well").fetchone()[0]
        self.n_perturbations = self.con.execute(
            "SELECT COUNT(*) FROM perturbation"
        ).fetchone()[0]

        logger.info(
            f"Database stats: {self.n_plates:,} plates, "
            f"{self.n_wells:,} wells, {self.n_perturbations:,} perturbations"
        )

    def get_load_data_csv_path(
        self,
        source: str,
        batch: str,
        plate: str
    ) -> str:
        """
        Construct load_data.csv S3 path for a plate.

        Args:
            source: Source identifier (e.g., "source_1")
            batch: Batch identifier (e.g., "Batch1_20221004")
            plate: Plate identifier (e.g., "UL000109")

        Returns:
            S3 path to load_data.csv

        Example:
            >>> path = filter.get_load_data_csv_path("source_1", "Batch1_20221004", "UL000109")
            >>> print(path)
            s3://cellpainting-gallery/cpg0016-jump/source_1/workspace/load_data_csv/Batch1_20221004/UL000109/load_data.csv
        """
        return (
            f"s3://{self.S3_BUCKET}/{self.S3_PROJECT}/"
            f"{source}/workspace/load_data_csv/{batch}/{plate}/load_data.csv"
        )

    def filter_by_plate_type(
        self,
        plate_types: List[str],
        max_plates: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter plates by type.

        Args:
            plate_types: List of plate types (e.g., ['COMPOUND', 'ORF'])
            max_plates: Optional maximum number of plates to return

        Returns:
            DataFrame with columns: Metadata_Source, Metadata_Batch,
                                   Metadata_Plate, Metadata_PlateType

        Example:
            >>> df = filter.filter_by_plate_type(['COMPOUND'], max_plates=100)
            >>> print(len(df))  # Up to 100 plates
        """
        types_str = "','".join(plate_types)

        query = f"""
            SELECT
                Metadata_Source,
                Metadata_Batch,
                Metadata_Plate,
                Metadata_PlateType
            FROM plate
            WHERE Metadata_PlateType IN ('{types_str}')
            ORDER BY Metadata_Plate
        """

        if max_plates:
            query += f" LIMIT {max_plates}"

        df = self.con.execute(query).fetchdf()

        logger.info(
            f"Filtered to {len(df)} plates "
            f"(types: {', '.join(plate_types)})"
        )

        return df

    def filter_by_perturbation(
        self,
        modalities: List[str],
        max_plates: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter plates by perturbation modality.

        Args:
            modalities: List of modalities (e.g., ['compound', 'crispr'])
            max_plates: Optional maximum number of plates

        Returns:
            DataFrame with plate information
        """
        mods_str = "','".join(modalities)

        query = f"""
            SELECT DISTINCT
                p.Metadata_Source,
                p.Metadata_Batch,
                p.Metadata_Plate,
                p.Metadata_PlateType
            FROM plate p
            JOIN well w ON p.Metadata_Plate = w.Metadata_Plate
            JOIN perturbation pert ON w.Metadata_JCP2022 = pert.Metadata_JCP2022
            WHERE pert.Metadata_perturbation_modality IN ('{mods_str}')
            ORDER BY p.Metadata_Plate
        """

        if max_plates:
            query += f" LIMIT {max_plates}"

        df = self.con.execute(query).fetchdf()

        logger.info(
            f"Filtered to {len(df)} plates "
            f"(modalities: {', '.join(modalities)})"
        )

        return df

    def filter_by_compounds(
        self,
        jcp_ids: List[str],
        return_wells: bool = True
    ) -> pd.DataFrame:
        """
        Filter to specific compounds.

        Args:
            jcp_ids: List of JCP2022 identifiers
            return_wells: If True, return well-level data; else plate-level

        Returns:
            DataFrame with plate and well information

        Example:
            >>> compounds = ['JCP2022_033924', 'JCP2022_085227']
            >>> df = filter.filter_by_compounds(compounds)
            >>> # Get plates containing these compounds
            >>> plates = df['Metadata_Plate'].unique()
        """
        ids_str = "','".join(jcp_ids)

        if return_wells:
            query = f"""
                SELECT
                    p.Metadata_Source,
                    p.Metadata_Batch,
                    p.Metadata_Plate,
                    w.Metadata_Well,
                    w.Metadata_JCP2022,
                    p.Metadata_PlateType
                FROM plate p
                JOIN well w ON p.Metadata_Plate = w.Metadata_Plate
                WHERE w.Metadata_JCP2022 IN ('{ids_str}')
                ORDER BY p.Metadata_Plate, w.Metadata_Well
            """
        else:
            query = f"""
                SELECT DISTINCT
                    p.Metadata_Source,
                    p.Metadata_Batch,
                    p.Metadata_Plate,
                    p.Metadata_PlateType
                FROM plate p
                JOIN well w ON p.Metadata_Plate = w.Metadata_Plate
                WHERE w.Metadata_JCP2022 IN ('{ids_str}')
                ORDER BY p.Metadata_Plate
            """

        df = self.con.execute(query).fetchdf()

        if return_wells:
            n_wells = len(df)
            n_plates = df['Metadata_Plate'].nunique()
            logger.info(
                f"Found {n_wells} wells across {n_plates} plates "
                f"for {len(jcp_ids)} compounds"
            )
        else:
            logger.info(f"Found {len(df)} plates for {len(jcp_ids)} compounds")

        return df

    def balanced_sample(
        self,
        n_per_type: int = 10,
        plate_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get balanced sample across plate types.

        Args:
            n_per_type: Number of plates per type
            plate_types: Optional list of types to include (default: all)

        Returns:
            DataFrame with sampled plates

        Example:
            >>> df = filter.balanced_sample(n_per_type=20)
            >>> # Get 20 plates from each type (COMPOUND, ORF, CRISPR, etc.)
        """
        if plate_types:
            types_str = "','".join(plate_types)
            where_clause = f"WHERE Metadata_PlateType IN ('{types_str}')"
        else:
            where_clause = ""

        query = f"""
            SELECT * FROM (
                SELECT
                    Metadata_Source,
                    Metadata_Batch,
                    Metadata_Plate,
                    Metadata_PlateType,
                    ROW_NUMBER() OVER (
                        PARTITION BY Metadata_PlateType
                        ORDER BY RANDOM()
                    ) as rn
                FROM plate
                {where_clause}
            )
            WHERE rn <= {n_per_type}
            ORDER BY Metadata_PlateType, Metadata_Plate
        """

        df = self.con.execute(query).fetchdf()
        df = df.drop(columns=['rn'])

        # Show distribution
        distribution = df.groupby('Metadata_PlateType').size()
        logger.info(f"Balanced sample distribution:\n{distribution}")

        return df

    def get_wells_for_plates(
        self,
        plate_ids: List[str],
        filter_jcp_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get well-level data for specific plates.

        Args:
            plate_ids: List of plate identifiers
            filter_jcp_ids: Optional list of JCP2022 IDs to filter wells

        Returns:
            DataFrame with well information
        """
        plates_str = "','".join(plate_ids)

        query = f"""
            SELECT
                p.Metadata_Source,
                p.Metadata_Batch,
                p.Metadata_Plate,
                w.Metadata_Well,
                w.Metadata_JCP2022
            FROM plate p
            JOIN well w ON p.Metadata_Plate = w.Metadata_Plate
            WHERE p.Metadata_Plate IN ('{plates_str}')
        """

        if filter_jcp_ids:
            ids_str = "','".join(filter_jcp_ids)
            query += f" AND w.Metadata_JCP2022 IN ('{ids_str}')"

        query += " ORDER BY p.Metadata_Plate, w.Metadata_Well"

        df = self.con.execute(query).fetchdf()

        logger.info(
            f"Found {len(df)} wells across {df['Metadata_Plate'].nunique()} plates"
        )

        return df

    def get_download_manifest(
        self,
        plates_df: pd.DataFrame,
        include_csv_paths: bool = True
    ) -> pd.DataFrame:
        """
        Generate download manifest with S3 paths.

        Args:
            plates_df: DataFrame from filter methods
            include_csv_paths: Add load_data.csv S3 paths

        Returns:
            DataFrame with download information
        """
        if include_csv_paths:
            plates_df = plates_df.copy()
            plates_df['load_data_csv_path'] = plates_df.apply(
                lambda row: self.get_load_data_csv_path(
                    row['Metadata_Source'],
                    row['Metadata_Batch'],
                    row['Metadata_Plate']
                ),
                axis=1
            )

        logger.info(f"Generated download manifest for {len(plates_df)} plates")

        return plates_df

    def get_plate_type_distribution(self) -> pd.DataFrame:
        """
        Get distribution of plate types in database.

        Returns:
            DataFrame with counts per plate type
        """
        query = """
            SELECT
                Metadata_PlateType,
                COUNT(*) as count
            FROM plate
            GROUP BY Metadata_PlateType
            ORDER BY count DESC
        """

        return self.con.execute(query).fetchdf()

    def get_perturbation_distribution(self) -> pd.DataFrame:
        """
        Get distribution of perturbation modalities.

        Returns:
            DataFrame with counts per modality
        """
        query = """
            SELECT
                Metadata_perturbation_modality,
                COUNT(*) as count
            FROM perturbation
            GROUP BY Metadata_perturbation_modality
            ORDER BY count DESC
        """

        return self.con.execute(query).fetchdf()

    def search_compounds_by_smiles(
        self,
        smiles_pattern: str,
        max_results: int = 1000
    ) -> pd.DataFrame:
        """
        Search compounds by SMILES pattern (substring match).

        Note: For chemical substructure search, use RDKit separately.

        Args:
            smiles_pattern: SMILES substring to search for
            max_results: Maximum number of results

        Returns:
            DataFrame with matching compounds
        """
        query = f"""
            SELECT
                Metadata_JCP2022,
                Metadata_SMILES,
                Metadata_InChIKey
            FROM compound
            WHERE Metadata_SMILES LIKE '%{smiles_pattern}%'
            LIMIT {max_results}
        """

        df = self.con.execute(query).fetchdf()

        logger.info(f"Found {len(df)} compounds matching SMILES pattern")

        return df

    def close(self):
        """Close database connection."""
        if hasattr(self, 'con'):
            self.con.close()
            logger.info("Closed database connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
