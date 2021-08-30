#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ashutosh

Initial python setup file for given data and its analysis.
"""

import numpy
import pandas
import geopandas

import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8})

# first survey CSV file with equipment testing data
survey1_file = "./data/yavatmal_rapid_pond_survey1_corrected.csv"
csv_yavatmal = pandas.read_csv(survey1_file, delimiter=";")
csv_yavatmal = csv_yavatmal.round({"pond_area": 0})
csv_yavatmal["pond_area"] = csv_yavatmal["pond_area"].astype(int)
csv_yv_uq_area = csv_yavatmal["pond_area"].unique()
csv_yv_uq_frequency = csv_yavatmal["pond_area"].value_counts(normalize=True, sort=False, dropna=False)


# second suvery CSV file focused on Ralegaon area
survey2_file = "./data/YavatmalRapidPond_survey2_corrected.csv"
csv_ralegaon = pandas.read_csv(survey2_file, delimiter=",", dtype={"lat": numpy.float32, "long": numpy.float32})
csv_ralegaon = csv_ralegaon.round({"area": 0})
csv_ralegaon["area"] = csv_ralegaon["area"].astype(int)
csv_rg_uq_area = csv_ralegaon["area"].unique()
csv_rg_uq_frequency = csv_ralegaon["area"].value_counts(normalize=True, sort=False, dropna=False)


# all water bodies in Yavatmal district from overpass-turbo.eu
osm_yavatmal_file = "./data/yavatamal_all_water_bodies.geojson"
osm_yavatmal = geopandas.read_file(osm_yavatmal_file)
osm_yavatmal = osm_yavatmal[osm_yavatmal["water"] != "river"]  # dropping "river" noise
osm_yavatmal["centroid"] = osm_yavatmal["geometry"].centroid
osm_yavatmal = osm_yavatmal.to_crs({"proj": "cea"})
osm_yavatmal["area"] = osm_yavatmal["geometry"].area  # /(10**6)  # if want area in sq.km.
# osm_yavatmal["perimeter"] = osm_yavatmal["geometry"].length  # /(10**3)
osm_yavatmal["area"] = osm_yavatmal["area"].astype(int)
oms_yv_uq_area = osm_yavatmal["area"].unique()
osm_yv_uq_frequency = osm_yavatmal["area"].value_counts(normalize=True, sort=False, dropna=False)


# all water bodies in Ralegaon area from overpass-turbo.eu
osm_ralegaon_file = "./data/ralegaon_all_water_bodies.geojson"
osm_ralegaon = geopandas.read_file(osm_ralegaon_file)
osm_ralegaon = osm_ralegaon[osm_ralegaon["water"] != "river"]  # dropping "river" noise
osm_ralegaon["centroid"] = osm_ralegaon["geometry"].centroid
osm_ralegaon = osm_ralegaon.to_crs({"proj": "cea"})
osm_ralegaon["area"] = osm_ralegaon["geometry"].area  # /(10**6)  # if want area in sq.km.
# osm_ralegaon["perimeter"] = osm_ralegaon["geometry"].length  # /(10**3)
osm_ralegaon["area"] = osm_ralegaon["area"].astype(int)
oms_rg_uq_area = osm_ralegaon["area"].unique()
osm_rg_uq_frequency = osm_ralegaon["area"].value_counts(normalize=True, sort=False, dropna=False)


# plotting latitude and longitude of water bodies
plt.figure(dpi=120)
osm_ralegaon["centroid"].plot(markersize=15, color="blue", ax=plt.gca())
csv_ralegaon.plot.scatter("long", "lat", s=15, color="red", ax=plt.gca())
plt.title("Ralegaon OSM and Survey Lat/Lon")
plt.xlabel("Longitude")
plt.ylabel("Latitutde")
plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
plt.legend(["OSM", "Survey"], loc="upper right", bbox_to_anchor=(1.5, 1))
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig("./plots_results/ralegaon_osm_survey_lat_lon.png", dpi=1200, facecolor=None, edgecolor=None,
            orientation="portrait", format="png", transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None)
plt.show()


# plotting latitude and longitude of water bodies
plt.figure(dpi=120)
osm_yavatmal["centroid"].plot(markersize=15, color="blue", ax=plt.gca())
csv_ralegaon.plot.scatter("long", "lat", s=15, color="red", ax=plt.gca())
plt.title("Yavatmal OSM and Survey Lat/Lon")
plt.xlabel("Longitude")
plt.ylabel("Latitutde")
plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
plt.legend(["OSM", "Survey-2"], loc="upper right", bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.savefig("./plots_results/yavatmal_osm_survey_lat_lon.png", dpi=1200, facecolor="w", edgecolor="w",
            orientation="portrait", format="png", transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None)
plt.show()


# plotting latitude and longitude of water bodies
plt.figure(dpi=120)
osm_yavatmal["centroid"].plot(markersize=15, color="blue", ax=plt.gca())
csv_ralegaon.plot.scatter("long", "lat", s=15, color="red", ax=plt.gca())
csv_yavatmal.plot.scatter("_Location of Pond_longitude", "_Location of Pond_latitude",
                          s=15, color="green", ax=plt.gca())
plt.title("Yavatmal OSM and Surveys Lat/Lon")
plt.xlabel("Longitude")
plt.ylabel("Latitutde")
plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
plt.legend(["OSM", "Survey-2", "Survey-1"], loc="upper right", bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.savefig("./plots_results/yavatmal_osm_surveys_lat_lon.png", dpi=1200, facecolor="w", edgecolor="w",
            orientation="portrait", format="png", transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None)
plt.show()


# plotting area distribution
plt.figure(dpi=120)
plt.scatter(oms_yv_uq_area, osm_yv_uq_frequency, s=15, color="blue", label="OSM Yavatmal")
plt.scatter(csv_rg_uq_area, csv_rg_uq_frequency, s=15, color="red", label="Survey-2")
plt.scatter(csv_yv_uq_area, csv_yv_uq_frequency, s=15, color="green", label="Survey-1")
plt.title("Yavatmal OSM and Surveys Area")
plt.xlabel("Area [Sq.mt.]")
plt.ylabel("Frequency [Normalized]")
plt.xscale("log")
plt.ylim(-0.1, 0.3)
plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
plt.tight_layout()
plt.savefig("./plots_results/yavatmal_osm_surveys_areas.png", dpi=1200, facecolor="w", edgecolor="w",
            orientation="portrait", format="png", transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None)
plt.show()


# coordinate conversion test script
# crs_system = [{"proj": "cea"}, {"init": "epsg:3857"}, {"init": "epsg:32633"}, {"init": "epsg:3395"}, {"init": "epsg:6933"}]
# CEA coordinate system
# osm_yavatmal = osm_yavatmal.to_crs({"proj": "cea"})

# for onecrs in crs_system:
#     print(onecrs)
#     data_proj = osm_yavatmal.to_crs(onecrs)
#     data_proj["area"] = data_proj["geometry"].area  # /(10**6)
#     data_proj["perimeter"] = data_proj["geometry"].length  # /(10**3)
#     print(data_proj["area"])
#     print(data_proj["perimeter"])
#     print("\n")

print("\n\nDone\n\n")
