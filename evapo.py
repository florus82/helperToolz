import datetime

class LandsatETFileManager:
    '''
    takes a list of paths to landsat provisional ET files, and helps to extract files filtered by a date, a date range, or by year, month, or day
    get_by_date: provide a date (e.g. datetime.date(2022,6,25))
    get_by_range: provide a starting and ending date (both dates will be included in the output)
    get_by_year: provide a year (int)
    get_by_month: provide a month (int, e.g. January = 1)
    get_by_day: provide a day (int)
    get_by_year_and_month: provide as 2 int
    get_all_dates: returns all dates of the initialiszed class. If no parameter provided, unsorted list will be returned. If boolean True provided, sorted list will be returned
    get_all_dates_by_year: provide a year, and all dates are returned (unique=T/F)
    '''
    def __init__(self, landsat_files):
        self.date_to_files = {}

        self._build_index(landsat_files)
    # internal function that fills dict when class is initialized
    def _build_index(self, landsat_files):
        for file in landsat_files:
            filename = file.split('/')[-1]
            year = int(filename[10:14])
            month = int(filename[14:16])
            day = int(filename[16:18])
            date = datetime.date(year, month, day)

            if date not in self.date_to_files:
                self.date_to_files[date] = []
            self.date_to_files[date].append(file)
    
    def get_by_date(self, date):
        return self.date_to_files.get(date, [])

    def get_by_range(self, start_date, end_date):
        matching_files = []
        for date, files in self.date_to_files.items():
            if start_date <= date <= end_date:
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
    
    def get_by_year(self, year):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.year == year: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
    
    def get_by_month(self, month):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.month == month: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
    
    def get_by_day(self, day):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.day == day: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
        
    def get_all_dates(self, sorted_output=False):
        dates = self.date_to_files.keys()
        return sorted(dates) if sorted_output else list(dates)
    
    def get_all_dates_by_year(self, year, unique=False):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.year == year: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        dates = [date for date, file in matching_files]
        if unique:
            dates  = list(dict.fromkeys(dates))
        return dates
    
    def get_by_year_and_month(self, year, month):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.year == year and date.month == month: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]