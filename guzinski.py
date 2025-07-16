import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *
from helperToolz.evapo import *

workhorse = True

if workhorse:
    origin = 'Aldhani/eoagritwin/'
else:
    origin = ''

import re
from other_repos.pyDMS.pyDMS.pyDMS import *

int_to_Month = {
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

dayCount_leap = [31,29,31,30,31,30,31,31,30,31,30,31]
dayCount_noleap = [31,28,31,30,31,30,31,31,30,31,30,31]