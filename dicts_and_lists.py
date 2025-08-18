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

MONTH_TO_02D = {v: k for k, v in INT_TO_MONTH.items()}

DAYCOUNT_LEAP = [31,29,31,30,31,30,31,31,30,31,30,31]
DAYCOUNT_NOLEAP = [31,28,31,30,31,30,31,31,30,31,30,31]

DRY_ADIABAT = 0.0098 # K/m
STANDARD_ADIABAT = 0.0065 # K/m
MOIST_ADIABAT = 0.002 # K/m

GRAVITY = 9.80665 # m/s**2
MOLAR_MASS_AIR = 0.0289644 # kg/mol
UNIVERSAL_GAS = 8.3144598 # J/(mol*K)