import itertools

import s2sphere
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


def decodeTrajectories(trajectories, scaler):
    trajectories = (trajectories * 127.5) + 127.5
    for x in range(len(trajectories)):
        trajectories[x] = scaler.inverse_transform(trajectories[x])
    s_c_id = list(itertools.chain(*trajectories))
    cellId = []
    for i in range(0, len(s_c_id)):
        cellId.append(s_c_id[i][0])
    cellId = list(map(int, cellId))
    map_lat = []
    map_lng = []
    for i in range(0, len(s_c_id)):
        ll = str(s2sphere.CellId(cellId[i]).to_lat_lng())
        latlng = ll.split(',', 1)
        lat = latlng[0].split(':', 1)
        map_lat.append(float(lat[1]))
        map_lng.append(float(latlng[1]))
    return map_lat, map_lng
