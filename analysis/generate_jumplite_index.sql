INSTALL httpfs;
LOAD httpfs;
SET enable_progress_bar = true;
.cd /work/datasets/jump_lite/misc/
-- replace this table for a list of all plates
CREATE TABLE jl_plates AS (SELECT Metadata_Plate FROM 'unique_plates.csv');
-- CREATE TABLE jump_jl_plates AS (SELECT DISTINCT Metadata_Source,Metadata_Batch,Metadata_Plate FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') NATURAL JOIN jl_plates);

-- encode redlisting
COPY (SELECT COLUMNS('^Metadata_(Source|Batch|Plate)$') FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') WHERE
((Metadata_Source = 'source_4' AND suffix(Metadata_Batch, 'Batch12')) OR
(Metadata_Source = 'source_3' AND (regexp_matches(Metadata_Batch, '^CP_3[23456]_all_Phenix1$') OR Metadata_Batch in ['CP59', 'CP60'])))) TO 'redlist.csv';

CREATE TABLE loaddata_uris AS (SELECT *, format('s3://cellpainting-gallery/cpg0016-jump/{}/workspace/load_data_csv/{}/{}/load_data_with_illum.csv', Metadata_Source, Metadata_Batch, Metadata_Plate) AS uri FROM jump_plates);
SET VARIABLE csv_files = (SELECT list(uri) FROM loaddata_uris);
CREATE OR REPLACE TABLE loaddata AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate|Well|Site)$'),COLUMNS('URL_Orig(DNA|Mito|AGP|ER|RNA)')  FROM read_csv(getVariable('csv_files'), union_by_name=True));
CREATE OR REPLACE TABLE unpivoted AS (UNPIVOT loaddata ON COLUMNS('URL_*') INTO NAME Metadata_Channel VALUE uri);


COPY loaddata TO 'jump_index.parquet';
COPY unpivoted TO 'jump_index_tidy.parquet';

COPY (UNPIVOT (SELECT B.* FROM 'metadata_dataset_filtered_4reps.parquet' AS A NATURAL JOIN (FROM 'jump_index.parquet') AS B) ON COLUMNS('URL_*') INTO NAME 'Metadata_Channel' VALUE 'uri') TO 'jl_index_tidy.parquet';
