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

drop SCHEMA master CASCADE;

CREATE SCHEMA master;

--
-- Name: image_attributes; Type: TABLE; Schema: public; Owner: kelvin
--


CREATE TABLE master.image_attributes (
   id text NOT NULL,
   aircraftplatform text,
   altitudecode text,
   area_or_point text,
   begin_date text,
   calibrationname text,
   calibrationversion text,
   completiondate TIMESTAMP,
   creationdate TIMESTAMP,
   datausersguidesource text,
   data_quality text,
   data_set text,
   datum text,
   day_night_flag char,
   end_date TIMESTAMP,
   experimentname text,
   flightcomment text,
   flightdate text,
   flightlinecomment text,
   flightlinenumber smallint,
   flightnumber text,
   geographicarea text,
   geolocationprocess text,
   granule_size integer,
   granule_version text,
   lat_ll real,
   lat_lr real,
   lat_ul real,
   lat_ur real,
   locationcode text,
   lon_ll real,
   lon_lr real,
   lon_ul real,
   lon_ur real,
   metadata_version text,
   mgeocontrolfilepath text,
   navdatapath text,
   navdatasource text,
   navformatcode text,
   navigational_precision text,
   other_aircraft_sensors text,
   principal_investigator text,
   producer_granule_id text,
   reference text,
   runtime text,
   scale_factor real[],
   sitelinerun text,
   softwareversion text,
   srf_dataset text,
   srf_ftp text,
   title text,
   totalflightlines text,
   units text,
   _fillvalue real
);

CREATE TABLE master.images (
  id text not null,
  geom public.geometry(geometry,4326) not null,
  "time" date not null,
  analoggain text,
  analogoffset text,
  blackbody1counts text,
  blackbody2counts text,
  calibrateddata text,
  calibrateddata_geo text,
  calibrationintercept text,
  calibrationslope text,
  datasetheader text,
  head1counts text,
  head2counts text,
  original text,
  pixelelevation text,
  pixellatitude text,
  pixellongitude text,
  sensorazimuthangle text,
  sensorzenithangle text,
  solarazimuthangle text,
  solarzenithangle text
);

COMMENT ON COLUMN master.images.master IS 'Original HDF file given from master';

ALTER TABLE ONLY master.image_attributes
    ADD CONSTRAINT master_image_attributes_pkey PRIMARY KEY (id);

ALTER TABLE ONLY master.images
    ADD CONSTRAINT master_images_pkey PRIMARY KEY (id);

CREATE INDEX idx_images_geom ON master.images USING gist (geom);
