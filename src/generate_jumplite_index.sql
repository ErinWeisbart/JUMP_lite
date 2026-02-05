INSTALL httpfs;
LOAD httpfs;
SET enable_progress_bar = true;
CREATE TABLE jl_plates AS (SELECT Metadata_Plate FROM '/work/datasets/jump_lite/misc/unique_plates.csv');
CREATE TABLE jump_jl_plates AS (SELECT DISTINCT Metadata_Source,Metadata_Batch,Metadata_Plate FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') NATURAL JOIN jl_plates);
CREATE TABLE loaddata_uris AS (SELECT *, format('s3://cellpainting-gallery/cpg0016-jump/{}/workspace/load_data_csv/{}/{}/load_data_with_illum.csv', Metadata_Source, Metadata_Batch, Metadata_Plate) AS uri FROM jump_jl_plates);
SET VARIABLE csv_files = (SELECT list(uri) FROM loaddata_uris);
CREATE OR REPLACE TABLE loaddata AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate|Well|Site)$'),COLUMNS('URL_Orig(DNA|Mito|AGP|ER|RNA)')  FROM read_csv(getVariable('csv_files'), union_by_name=True));
CREATE OR REPLACE TABLE unpivoted AS (UNPIVOT loaddata ON COLUMNS('URL_*') INTO NAME Metadata_Channel VALUE uri);

COPY loaddata TO '/work/datasets/jump_lite/misc/jump_index.parquet';
COPY unpivoted TO '/work/datasets/jump_lite/misc/jump_index_tidy.parquet';
