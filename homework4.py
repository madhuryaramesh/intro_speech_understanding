def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''

    future_birthdays = []

    for birthday in birthdays:
        if birthday > date:
            future_birthdays.append(birthday)

    if future_birthdays:
        birthday = min(future_birthdays)
    else:
        birthday = min(birthdays)

    list_of_names = birthdays[birthday]

    return birthday, list_of_names