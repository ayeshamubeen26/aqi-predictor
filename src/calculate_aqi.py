def calculate_pm25_aqi(pm25):
    """
    Converts a PM2.5 concentration (µg/m³) into a US EPA AQI value,
    using the breakpoints updated in May 2024.
    """
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ]

    pm25 = round(pm25, 1)

    for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
        if conc_low <= pm25 <= conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (pm25 - conc_low) + aqi_low
            return round(aqi)

    # Above the highest breakpoint — cap at 500
    return 500

def calculate_pm10_aqi(pm10):
    """
    Converts a PM10 concentration (µg/m³) into a US EPA AQI value.
    """
    breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ]

    pm10 = round(pm10)

    for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
        if conc_low <= pm10 <= conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (pm10 - conc_low) + aqi_low
            return round(aqi)

    return 500

def calculate_aqi(pm25, pm10):
    """
    Returns the overall AQI: the worse (higher) of the PM2.5 and PM10
    sub-indices, matching how the EPA determines the dominant pollutant.
    """
    aqi_pm25 = calculate_pm25_aqi(pm25)
    aqi_pm10 = calculate_pm10_aqi(pm10)
    return max(aqi_pm25, aqi_pm10)

if __name__ == "__main__":
    # Quick sanity checks against known reference values
    print("PM2.5 = 20 ->", calculate_pm25_aqi(20), "(expect ~68)")
    print("PM2.5 = 50 ->", calculate_pm25_aqi(50), "(expect ~137)")
    print("PM10 = 100 ->", calculate_pm10_aqi(100), "(expect ~73)")
    print("Combined (pm2.5=50, pm10=100) ->", calculate_aqi(50, 100))