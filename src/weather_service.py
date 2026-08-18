"""
ExplainCrop-AI: Weather Service
Integrates with Open-Meteo API for real-time global weather and rainfall estimation.
"""

import requests
from typing import Dict, Any, Optional


def get_weather_by_coordinates(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetches real-time temperature, humidity, and rainfall estimates for given coordinates.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current=temperature_2m,relative_humidity_2m,precipitation,rain&"
            f"daily=precipitation_sum&timezone=auto"
        )
        response = requests.get(url, timeout=6)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        temperature = current.get("temperature_2m", 25.0)
        humidity = current.get("relative_humidity_2m", 70.0)

        # Estimate average annual/seasonal rainfall in mm equivalent for crop modeling
        # Open-Meteo returns daily precipitation sum (mm).
        # We estimate seasonal benchmark based on recent precipitation patterns
        daily_precip = daily.get("precipitation_sum", [5.0])
        recent_rain_mean = sum(daily_precip) / max(len(daily_precip), 1)
        # Scaled representative seasonal rainfall parameter
        estimated_seasonal_rainfall = max(30.0, round(recent_rain_mean * 25.0 + 50.0, 1))

        return {
            "success": True,
            "latitude": lat,
            "longitude": lon,
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "rainfall": estimated_seasonal_rainfall,
            "units": {
                "temperature": "°C",
                "humidity": "%",
                "rainfall": "mm",
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "temperature": 25.0,
            "humidity": 70.0,
            "rainfall": 100.0,
        }


def get_weather_by_city(city_name: str) -> Dict[str, Any]:
    """
    Geocodes city name and retrieves current weather data.
    """
    if not city_name or not city_name.strip():
        return {"success": False, "error": "City name cannot be empty"}

    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name.strip()}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=5)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        results = geo_data.get("results")
        if not results or len(results) == 0:
            return {"success": False, "error": f"City '{city_name}' not found."}

        first_res = results[0]
        lat = first_res.get("latitude")
        lon = first_res.get("longitude")
        name = first_res.get("name")
        country = first_res.get("country", "")
        admin1 = first_res.get("admin1", "")

        weather = get_weather_by_coordinates(lat, lon)
        weather["location_name"] = f"{name}, {admin1 + ', ' if admin1 else ''}{country}".strip(
            ", "
        )
        return weather

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    test_cities = ["Coimbatore", "London", "Nairobi"]
    for c in test_cities:
        res = get_weather_by_city(c)
        print(f"[{c}] -> {res}")
