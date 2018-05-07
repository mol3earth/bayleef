--
-- PostgreSQL database dump
--

-- Dumped from database version 9.6.8
-- Dumped by pg_dump version 9.6.8

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: tiger; Type: SCHEMA; Schema: -; Owner: kelvin
--

CREATE SCHEMA tiger;


--
-- Name: tiger_data; Type: SCHEMA; Schema: -; Owner: kelvin
--

CREATE SCHEMA tiger_data;

--
-- Name: topology; Type: SCHEMA; Schema: -; Owner: kelvin
--

CREATE SCHEMA topology;

--
-- Name: SCHEMA topology; Type: COMMENT; Schema: -; Owner: kelvin
--

COMMENT ON SCHEMA topology IS 'PostGIS Topology schema';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner:
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: fuzzystrmatch; Type: EXTENSION; Schema: -; Owner:
--

CREATE EXTENSION IF NOT EXISTS fuzzystrmatch WITH SCHEMA public;


--
-- Name: EXTENSION fuzzystrmatch; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION fuzzystrmatch IS 'determine similarities and distance between strings';


--
-- Name: postgis; Type: EXTENSION; Schema: -; Owner:
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry, geography, and raster spatial types and functions';


--
-- Name: postgis_tiger_geocoder; Type: EXTENSION; Schema: -; Owner:
--

CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder WITH SCHEMA tiger;


--
-- Name: EXTENSION postgis_tiger_geocoder; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION postgis_tiger_geocoder IS 'PostGIS tiger geocoder and reverse geocoder';


--
-- Name: postgis_topology; Type: EXTENSION; Schema: -; Owner:
--

CREATE EXTENSION IF NOT EXISTS postgis_topology WITH SCHEMA topology;


--
-- Name: EXTENSION postgis_topology; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION postgis_topology IS 'PostGIS topology spatial types and functions';


SET default_tablespace = '';

SET default_with_oids = false;

CREATE SCHEMA landsat_8_c1;

--
-- Name: image_attributes; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.image_attributes (
    cloud_cover bigint,
    cloud_cover_land bigint,
    earth_sun_distance double precision,
    image_quality_oli bigint,
    image_quality_tirs bigint,
    landsat_scene_id text NOT NULL,
    roll_angle double precision,
    saturation_band_1 text,
    saturation_band_2 text,
    saturation_band_3 text,
    saturation_band_4 text,
    saturation_band_5 text,
    saturation_band_6 text,
    saturation_band_7 text,
    saturation_band_8 text,
    saturation_band_9 text,
    sun_azimuth double precision,
    sun_elevation double precision,
    tirs_ssm_model text,
    ground_control_points_version  double precision,
    ground_control_points_model  double precision,
    geometric_rmse_model  double precision,
    geometric_rmse_model_y  double precision,
    geometric_rmse_model_x  double precision,
    tirs_ssm_position_status text,
    tirs_stray_light_correction_source text,
    truncation_oli text,
    is_daytime boolean
);



--
-- Name: images; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.images (
    landsat_scene_id text NOT NULL,
    geom public.geometry(Geometry,4326) NOT NULL,
    "time" date NOT NULL,
    b1 text,
    b2 text,
    b3 text,
    b4 text,
    b5 text,
    b6 text,
    b7 text,
    b8 text,
    b9 text,
    b10 text,
    b11 text,
    bqa text,
    metafile text,
    ang text
);


--
-- Name: metadata_file_info; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.metadata_file_info (
    collection_number bigint,
    file_date text,
    landsat_product_id text,
    landsat_scene_id text NOT NULL,
    origin text,
    processing_software_version text,
    request_id text,
    station_id text
);

--
-- Name: min_max_pixel_value; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.min_max_pixel_value (
    landsat_scene_id text NOT NULL,
    quantize_cal_max_band_1 bigint,
    quantize_cal_max_band_10 bigint,
    quantize_cal_max_band_11 bigint,
    quantize_cal_max_band_2 bigint,
    quantize_cal_max_band_3 bigint,
    quantize_cal_max_band_4 bigint,
    quantize_cal_max_band_5 bigint,
    quantize_cal_max_band_6 bigint,
    quantize_cal_max_band_7 bigint,
    quantize_cal_max_band_8 bigint,
    quantize_cal_max_band_9 bigint,
    quantize_cal_min_band_1 bigint,
    quantize_cal_min_band_10 bigint,
    quantize_cal_min_band_11 bigint,
    quantize_cal_min_band_2 bigint,
    quantize_cal_min_band_3 bigint,
    quantize_cal_min_band_4 bigint,
    quantize_cal_min_band_5 bigint,
    quantize_cal_min_band_6 bigint,
    quantize_cal_min_band_7 bigint,
    quantize_cal_min_band_8 bigint,
    quantize_cal_min_band_9 bigint
);


--
-- Name: min_max_radiance; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.min_max_radiance (
    landsat_scene_id text NOT NULL,
    radiance_maximum_band_1 double precision,
    radiance_maximum_band_10 double precision,
    radiance_maximum_band_11 double precision,
    radiance_maximum_band_2 double precision,
    radiance_maximum_band_3 double precision,
    radiance_maximum_band_4 double precision,
    radiance_maximum_band_5 double precision,
    radiance_maximum_band_6 double precision,
    radiance_maximum_band_7 double precision,
    radiance_maximum_band_8 double precision,
    radiance_maximum_band_9 double precision,
    radiance_minimum_band_1 double precision,
    radiance_minimum_band_10 double precision,
    radiance_minimum_band_11 double precision,
    radiance_minimum_band_2 double precision,
    radiance_minimum_band_3 double precision,
    radiance_minimum_band_4 double precision,
    radiance_minimum_band_5 double precision,
    radiance_minimum_band_6 double precision,
    radiance_minimum_band_7 double precision,
    radiance_minimum_band_8 double precision,
    radiance_minimum_band_9 double precision
);


--
-- Name: min_max_reflectance; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.min_max_reflectance (
    landsat_scene_id text NOT NULL,
    reflectance_maximum_band_1 double precision,
    reflectance_maximum_band_2 double precision,
    reflectance_maximum_band_3 double precision,
    reflectance_maximum_band_4 double precision,
    reflectance_maximum_band_5 double precision,
    reflectance_maximum_band_6 double precision,
    reflectance_maximum_band_7 double precision,
    reflectance_maximum_band_8 double precision,
    reflectance_maximum_band_9 double precision,
    reflectance_minimum_band_1 double precision,
    reflectance_minimum_band_2 double precision,
    reflectance_minimum_band_3 double precision,
    reflectance_minimum_band_4 double precision,
    reflectance_minimum_band_5 double precision,
    reflectance_minimum_band_6 double precision,
    reflectance_minimum_band_7 double precision,
    reflectance_minimum_band_8 double precision,
    reflectance_minimum_band_9 double precision
);


--
-- Name: product_metadata; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.product_metadata (
    angle_coefficient_file_name text,
    bpf_name_oli text,
    bpf_name_tirs text,
    collection_category text,
    corner_ll_lat_product double precision,
    corner_ll_lon_product double precision,
    corner_ll_projection_x_product double precision,
    corner_ll_projection_y_product double precision,
    corner_lr_lat_product double precision,
    corner_lr_lon_product double precision,
    corner_lr_projection_x_product double precision,
    corner_lr_projection_y_product double precision,
    corner_ul_lat_product double precision,
    corner_ul_lon_product double precision,
    corner_ul_projection_x_product double precision,
    corner_ul_projection_y_product double precision,
    corner_ur_lat_product double precision,
    corner_ur_lon_product double precision,
    corner_ur_projection_x_product double precision,
    corner_ur_projection_y_product double precision,
    cpf_name text,
    data_type text,
    date_acquired date,
    elevation_source text,
    file_name_band_1 text,
    file_name_band_10 text,
    file_name_band_11 text,
    file_name_band_2 text,
    file_name_band_3 text,
    file_name_band_4 text,
    file_name_band_5 text,
    file_name_band_6 text,
    file_name_band_7 text,
    file_name_band_8 text,
    file_name_band_9 text,
    file_name_band_quality text,
    landsat_scene_id text NOT NULL,
    metadata_file_name text,
    nadir_offnadir text,
    output_format text,
    panchromatic_lines bigint,
    panchromatic_samples bigint,
    reflective_lines bigint,
    reflective_samples bigint,
    rlut_file_name text,
    scene_center_time text,
    sensor_id text,
    spacecraft_id text,
    target_wrs_path bigint,
    target_wrs_row bigint,
    thermal_lines bigint,
    thermal_samples bigint,
    wrs_path bigint,
    wrs_row bigint
);


--
-- Name: projection_parameters; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.projection_parameters (
    datum text,
    ellipsoid text,
    grid_cell_size_panchromatic double precision,
    grid_cell_size_reflective double precision,
    grid_cell_size_thermal double precision,
    landsat_scene_id text NOT NULL,
    map_projection text,
    orientation text,
    resampling_option text,
    utm_zone bigint
);


--
-- Name: radiometric_rescaling; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.radiometric_rescaling (
    landsat_scene_id text NOT NULL,
    radiance_add_band_1 double precision,
    radiance_add_band_10 double precision,
    radiance_add_band_11 double precision,
    radiance_add_band_2 double precision,
    radiance_add_band_3 double precision,
    radiance_add_band_4 double precision,
    radiance_add_band_5 double precision,
    radiance_add_band_6 double precision,
    radiance_add_band_7 double precision,
    radiance_add_band_8 double precision,
    radiance_add_band_9 double precision,
    radiance_mult_band_1 double precision,
    radiance_mult_band_10 double precision,
    radiance_mult_band_11 double precision,
    radiance_mult_band_2 double precision,
    radiance_mult_band_3 double precision,
    radiance_mult_band_4 double precision,
    radiance_mult_band_5 double precision,
    radiance_mult_band_6 double precision,
    radiance_mult_band_7 double precision,
    radiance_mult_band_8 double precision,
    radiance_mult_band_9 double precision,
    reflectance_add_band_1 double precision,
    reflectance_add_band_2 double precision,
    reflectance_add_band_3 double precision,
    reflectance_add_band_4 double precision,
    reflectance_add_band_5 double precision,
    reflectance_add_band_6 double precision,
    reflectance_add_band_7 double precision,
    reflectance_add_band_8 double precision,
    reflectance_add_band_9 double precision,
    reflectance_mult_band_1 double precision,
    reflectance_mult_band_2 double precision,
    reflectance_mult_band_3 double precision,
    reflectance_mult_band_4 double precision,
    reflectance_mult_band_5 double precision,
    reflectance_mult_band_6 double precision,
    reflectance_mult_band_7 double precision,
    reflectance_mult_band_8 double precision,
    reflectance_mult_band_9 double precision
);


--
-- Name: tirs_thermal_constants; Type: TABLE; Schema: public; Owner: kelvin
--

CREATE TABLE landsat_8_c1.tirs_thermal_constants (
    k1_constant_band_10 double precision,
    k1_constant_band_11 double precision,
    k2_constant_band_10 double precision,
    k2_constant_band_11 double precision,
    landsat_scene_id text NOT NULL
);


--
-- Name: image_attributes image_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.image_attributes
    ADD CONSTRAINT image_attributes_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: images images_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.images
    ADD CONSTRAINT images_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: metadata_file_info metadata_file_info_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.metadata_file_info
    ADD CONSTRAINT metadata_file_info_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: min_max_pixel_value min_max_pixel_value_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.min_max_pixel_value
    ADD CONSTRAINT min_max_pixel_value_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: min_max_radiance min_max_radiance_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.min_max_radiance
    ADD CONSTRAINT min_max_radiance_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: min_max_reflectance min_max_reflectance_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.min_max_reflectance
    ADD CONSTRAINT min_max_reflectance_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: product_metadata product_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.product_metadata
    ADD CONSTRAINT product_metadata_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: projection_parameters projection_parameters_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.projection_parameters
    ADD CONSTRAINT projection_parameters_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: radiometric_rescaling radiometric_rescaling_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.radiometric_rescaling
    ADD CONSTRAINT radiometric_rescaling_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: tirs_thermal_constants tirs_thermal_constants_pkey; Type: CONSTRAINT; Schema: public; Owner: kelvin
--

ALTER TABLE ONLY landsat_8_c1.tirs_thermal_constants
    ADD CONSTRAINT tirs_thermal_constants_pkey PRIMARY KEY (landsat_scene_id);


--
-- Name: idx_images_geom; Type: INDEX; Schema: public; Owner: kelvin
--

CREATE INDEX idx_images_geom ON landsat_8_c1.images USING gist (geom);


--
-- Name: images_dix; Type: INDEX; Schema: public; Owner: kelvin
--

CREATE INDEX images_dix ON landsat_8_c1.images USING btree ("time");


--
-- Name: images_gix; Type: INDEX; Schema: public; Owner: kelvin
--

CREATE INDEX images_gix ON landsat_8_c1.images USING gist (geom);


--
-- PostgreSQL database dump complete
--
