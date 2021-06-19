import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim


def fetchGeoLocation(cityCountry):
    """
    Using Nominatim OpenAPI to fetch Longitude and Latitude Data
    :return: [south Latitude
            north Latitude,
            west Longitude,
            east Longitude]
    :param: cityCountry : format string 'city, country code' ex. 'Lynon, France'
    """
    # HTTP Request recommended : "Application Name"
    app = Nominatim(user_agent="geoLife")
    location = app.geocode(cityCountry).raw
    # pprint(location) # pretty print JSON returned from OpenStreetMap dataset

    return location["boundingbox"]

def dropOutlyingData(df, boundingbox):
    """
    Remove data outside of bounding box longitude and latitude.
    :param: Dataframe with ['Longitude'] and ['Latitude'] column labels
    :param: Bounding Box : [min Lat, max Lat, min Lon, max Lon]
    """
    lat = boundingbox[0:2]
    lon = boundingbox[2:4]

    # Query data within bounds as forced floats
    return df.loc[
        (df.Longitude >= float(lon[0]))
        & (df.Longitude <= float(lon[1]))
        & (df.Latitude >= float(lat[0]))
        & (df.Latitude <= float(lat[1]))
    ].reset_index(drop=True)
