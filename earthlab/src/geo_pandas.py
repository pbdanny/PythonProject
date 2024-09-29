import geopandas
import geodatasets

from matplotlib import pyplot as plt

path_to_data = geodatasets.get_path("nybb")
gdf = geopandas.read_file(path_to_data)

gdf
gdf.geometry
gdf.crs
gdf.index
gdf = gdf.set_index("BoroName")

gdf.area
gdf.boundary
gdf["boundary"] = gdf.boundary
gdf.centroid

gdf["centroid"] = gdf.centroid
staten_cen = gdf["centroid"].iloc[0]
type(staten_cen)
gdf["distance"] = gdf["centroid"].distance(staten_cen)
gdf["distance"].mean()

gdf.plot(legend=True)
gdf.explore()

gdf = gdf.set_geometry("centroid")
gdf.plot()
ax = gdf["geometry"].plot()
gdf["centroid"].plot(ax=ax, color="black")

gdf = gdf.set_geometry("geometry")

ax = gdf.convex_hull.plot(alpha=0.5)
gdf["boundary"].plot(ax = ax, color="white", linewidth=0.5)

gdf["buffered"] = gdf.buffer(10000)
gdf["buffered_centroid"] = gdf["centroid"].buffer(10000)

ax = gdf["buffered"].plot(alpha=0.5)
gdf["buffered_centroid"].plot(ax=ax, color="red", alpha=0.5)
gdf["boundary"].plot(ax=ax, color="white", linewidth=0.5)

brooklyn = gdf.loc["Brooklyn", "geometry"]
brooklyn
type(brooklyn)

gdf["buffered"].intersects(brooklyn)

gdf["within"] = gdf["buffered_centroid"].within(gdf)
gdf["within"]

gdf = gdf.set_geometry("buffered_centroid")
ax = gdf.plot("within", legend=True,
              categorical=True, legend_kwds={"loc":"upper left"})
gdf["boundary"].plot(ax = ax, color="black", linewidth=0.5)

gdf = gdf.set_geometry("geometry")
boroughs_4326 = gdf.to_crs("EPSG:4326")
boroughs_4326.plot()
boroughs_4326.crs