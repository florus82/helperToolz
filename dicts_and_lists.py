INT_TO_MONTH = {
    '01': 'January',
    '02': 'February',
    '03': 'March',
    '04': 'April',
    '05': 'May',
    '06': 'June',
    '07': 'July',
    '08': 'August',
    '09': 'September',
    '10': 'October',
    '11': 'November',
    '12': 'December'
    }

REAL_INT_TO_MONTH = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
    }

MONTH_TO_02D = {v: k for k, v in INT_TO_MONTH.items()}

DAYCOUNT_LEAP = [31,29,31,30,31,30,31,31,30,31,30,31]
DAYCOUNT_NOLEAP = [31,28,31,30,31,30,31,31,30,31,30,31]

DRY_ADIABAT = 0.0098 # K/m
STANDARD_ADIABAT = 0.0065 # K/m
MOIST_ADIABAT = 0.002 # K/m

GRAVITY = 9.80665 # m/s**2
MOLAR_MASS_AIR = 0.0289644 # kg/mol
UNIVERSAL_GAS = 8.3144598 # J/(mol*K)

SOLAR_CONST = 1367.0 # Solar constant (W/m^2)


# for thuenen masking
VALID_AGRO_VALUES = [
    200, # Permanent grassland
    1101,# Winter wheat
    1102, # Winter barley
    1103, # Winter rye
    1201, # Spring barley
    1202, # Spring oat
    1300, # Maize
    1401, # Potato
    1402, # Sugar beet
    1501, # Winter rapeseed
    1502, # Sunflower
    1602, # Cultivated grassland
    1603, # Vegetables
    1611, # Peas
    1612, # Broad bean
    1613, # Lupin
    1614, # Soy
    #3001, # Small woody features
    3002, # Other agricultural areas
    3003, # Fallow land
    #3004, # Other areas
    #3011, # Small woody features on other land
    #4001, # Grapevine
    4002, # Hops
    4003, # Orchard
]


PLANCK = 6.62607015e-34      # Planck constant (J·s)
SPEEDL = 299792458.0         # speed of light (m/s)
BOLTZ = 1.380649e-23        # Boltzmann constant (J/K)

SLSTR_BANDS = {
    "S7": 3.742e-6,
    "S8": 10.854e-6,
    "S9": 12.023e-6
}