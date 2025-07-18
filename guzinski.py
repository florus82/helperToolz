from helperToolz.helpsters import is_leap_year
from helperToolz.dicts_and_lists import DAYCOUNT_LEAP, DAYCOUNT_NOLEAP, INT_TO_MONTH
import numpy as np

def transform_compositeDate_into_LSTbands(compDate, dayrange):
    """calculates which LST bands are associated with a compDate, based on dayrange
    A dictionary will be returned that contains the information on the month and band of each LST to be read-in
    Please note, that this benefits LST daily composites stored as monthly stacks, where the band of each stack corresponds to the day of that month

    Args:
        compDate (string): the date from the FORCE filesname in string in format YYYYMMDD
        dayrange (int): LST bands from how many days +/- from compdate should be considered
    """
    year = int(compDate[0:4])

    # what bands (days) in LST from which month are associated with the current S2 scene
    if is_leap_year(year):
        dayCountList = DAYCOUNT_LEAP
        end_day = 367
        add_to_april = 1 # needed because LST from +/-4 days away2 from S2 compos
    else:
        dayCountList = DAYCOUNT_NOLEAP
        end_day = 366
        add_to_april = 0

    day_sequence = [f'{year}{month:02d}{day:02d}' 
            for month, days_in_month in enumerate(dayCountList, start=1)
            for day in range(1, days_in_month + 1)]
        
    dicti = {}

    for i in range(1, end_day):
        month_idx = np.where(np.cumsum(dayCountList) >= i)[0][0]
        day_in_month =  i - (np.cumsum(dayCountList)[month_idx - 1] if month_idx > 0 else 0)

        dicti[i] = {
            'month': INT_TO_MONTH[f'{(month_idx + 1):02d}'],
            'band': day_in_month
        }

    index_in_sequence = [i+1 for i, day in enumerate(day_sequence) if day == compDate][0]

    # check for the special case, when compDate is YYYY0406 and dayrange == 4 as with this setting 1 April would not be processed
    # due to the FORCE processing with TSI at 9 day intervall that happened
    if (add_to_april == 1 and index_in_sequence == 97) or (add_to_april == 0 and index_in_sequence == 96):
        dayrange_low = dayrange + 1
        dayrange_up = dayrange
    else:
        dayrange_low = dayrange
        dayrange_up = dayrange


    return {k: v for k, v in dicti.items() if index_in_sequence - dayrange_low <= k <= index_in_sequence + dayrange_up}