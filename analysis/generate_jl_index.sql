.maxwidth 80
.maxrows 4
.cd /work/datasets/jump_lite/misc/
INSTALL httpfs;
LOAD httpfs;
CREATE OR REPLACE TABLE redlist AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate)$') FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') WHERE
((Metadata_Source = 'source_4' AND suffix(Metadata_Batch, 'Batch12')) OR
(Metadata_Source = 'source_3' AND (regexp_matches(Metadata_Batch, '^CP_3[23456]_all_Phenix1$') OR Metadata_Batch in ['CP59', 'CP60'])) OR
((Metadata_Source = 'source_15') AND Metadata_Plate in ['PEP00004458', 'PEP00004421'])));
COPY redlist TO 'redlist.csv';
FROM redlist;

CREATE OR REPLACE TABLE graylist AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate)$') FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') WHERE regexp_matches(Metadata_Plate, 'CP-CC9-R[123456]-28'));
COPY graylist TO 'graylist.csv';
FROM graylist;

CREATE OR REPLACE TABLE jump_plates AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate)$') FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') ANTI JOIN (FROM 'graylist.csv' UNION ALL FROM 'redlist.csv') using(Metadata_Source,Metadata_Batch,Metadata_Plate));
FROM jump_plates;

.maxwidth 80
CREATE OR REPLACE TABLE loaddata_uris AS (SELECT *, format('s3://cellpainting-gallery/cpg0016-jump/{}/workspace/load_data_csv/{}/{}/load_data_with_illum.csv', Metadata_Source, Metadata_Batch, Metadata_Plate) AS uri FROM jump_plates);
FROM loaddata_uris;
SET VARIABLE csv_files = (SELECT list(uri) FROM loaddata_uris);

CREATE OR REPLACE TYPE source_enum AS ENUM (SELECT DISTINCT Metadata_Source FROM loaddata_uris);
CREATE OR REPLACE TYPE batch_enum AS ENUM (SELECT DISTINCT Metadata_Batch FROM loaddata_uris);
CREATE OR REPLACE TYPE plate_enum AS ENUM (SELECT DISTINCT Metadata_Plate FROM loaddata_uris);
CREATE OR REPLACE TABLE jump_index AS (SELECT
    Metadata_Source::source_enum AS Metadata_Source,
    Metadata_Batch::batch_enum AS Metadata_Batch,
    Metadata_Plate::plate_enum AS Metadata_Plate,
    Metadata_Well,
    Metadata_Site,
  COLUMNS('URL_Orig(DNA|Mito|AGP|ER|RNA)'),
  FROM read_csv(getVariable('csv_files'), union_by_name=True));
  COPY jump_index TO 'jump_index.parquet' (COMPRESSION 'ZSTD');
FROM jump_index;

CREATE OR REPLACE TABLE jl_index AS (FROM jump_index JOIN (SELECT #2 AS Metadata_Plate FROM Read_csv('https://zenodo.org/api/records/18705140/files/jl_plates.csv/content')) using(Metadata_Plate));
COPY jl_index TO 'jl_index.parquet' (COMPRESSION 'ZSTD');
FROM jl_index;

CREATE OR REPLACE TABLE jl_index_sampled AS ( SELECT * EXCLUDE rn FROM (SELECT *, row_number() OVER (PARTITION BY Metadata_Plate,Metadata_Well) as rn FROM jl_index) WHERE rn < 5);
COPY jl_index_sampled TO 'jl_index_sampled.parquet' (COMPRESSION 'ZSTD');
FROM jl_index_sampled;
