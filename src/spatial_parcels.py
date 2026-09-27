"""
CropMind AI: Farm Spatial Parcel Geometry & Geodesic Field Area Engine
Computes polygon acreage, perimeter, centroid coordinates, and spatial field compactness from GPS boundaries.
"""

from typing import Dict, Any, List, Tuple, Optional
import math


def calculate_polygon_area_acres(coordinates: List[Tuple[float, float]]) -> float:
    """
    Computes geodesic polygon area in acres from a list of (lat, lon) coordinates
    using spherical excess projection (WGS84 ellipsoid approximation).
    """
    if len(coordinates) < 3:
        return 0.0
        
    # Radius of earth in meters
    R = 6378137.0
    
    # Convert lat/lon to radians
    coords_rad = [(math.radians(lat), math.radians(lon)) for lat, lon in coordinates]
    
    # Close polygon if not already closed
    if coords_rad[0] != coords_rad[-1]:
        coords_rad.append(coords_rad[0])
        
    area_sq_meters = 0.0
    num_pts = len(coords_rad)
    
    for i in range(num_pts - 1):
        lat1, lon1 = coords_rad[i]
        lat2, lon2 = coords_rad[i + 1]
        area_sq_meters += (lon2 - lon1) * (2.0 + math.sin(lat1) + math.sin(lat2))
        
    area_sq_meters = abs(area_sq_meters * (R ** 2) / 2.0)
    
    # 1 Acre = 4046.8564224 square meters
    area_acres = area_sq_meters / 4046.8564224
    return round(area_acres, 3)


def calculate_polygon_perimeter_meters(coordinates: List[Tuple[float, float]]) -> float:
    """Computes total perimeter boundary length in meters using Haversine formulation."""
    if len(coordinates) < 2:
        return 0.0
        
    R = 6378137.0
    total_dist = 0.0
    
    coords = list(coordinates)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
        
    for i in range(len(coords) - 1):
        lat1, lon1 = math.radians(coords[i][0]), math.radians(coords[i][1])
        lat2, lon2 = math.radians(coords[i+1][0]), math.radians(coords[i+1][1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        total_dist += R * c
        
    return round(total_dist, 1)


def calculate_polygon_centroid(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Computes arithmetic centroid of the polygon vertices."""
    if not coordinates:
        return (0.0, 0.0)
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    return (round(sum(lats) / len(lats), 6), round(sum(lons) / len(lons), 6))


def analyze_farm_parcel(
    parcel_name: str,
    boundary_coordinates: List[Tuple[float, float]],
    soil_type: str = "Clay Loam",
    primary_crop: Optional[str] = "Rice"
) -> Dict[str, Any]:
    """
    Performs complete spatial geometry evaluation for a farm boundary.
    """
    area_acres = calculate_polygon_area_acres(boundary_coordinates)
    perimeter_m = calculate_polygon_perimeter_meters(boundary_coordinates)
    centroid = calculate_polygon_centroid(boundary_coordinates)
    area_hectares = round(area_acres * 0.404686, 3)
    
    # Shape compactness index (Isoperimetric Quotient: 4 * pi * Area / Perimeter^2)
    area_sq_m = area_acres * 4046.856
    compactness = (4.0 * math.pi * area_sq_m) / (perimeter_m ** 2) if perimeter_m > 0 else 0.0
    compactness = round(min(1.0, max(0.1, compactness)), 3)

    return {
        "parcel_name": parcel_name,
        "primary_crop": primary_crop,
        "soil_type": soil_type,
        "vertices_count": len(boundary_coordinates),
        "centroid_lat_lon": {"latitude": centroid[0], "longitude": centroid[1]},
        "geometry": {
            "area_acres": area_acres,
            "area_hectares": area_hectares,
            "perimeter_meters": perimeter_m,
            "isoperimetric_compactness_score": compactness,
            "field_shape_classification": "Regular / High Workability" if compactness >= 0.55 else "Irregular / Narrow Strips"
        },
        "boundary_coordinates": [{"latitude": c[0], "longitude": c[1]} for c in boundary_coordinates]
    }
