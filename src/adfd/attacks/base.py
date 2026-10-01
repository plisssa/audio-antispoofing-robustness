ATTACKS = {}


def register_attack(name):
    def wrap(function):
        ATTACKS[name] = function
        return function

    return wrap


def get_attack(name):
    return ATTACKS[name]


def available_attacks():
    return sorted(ATTACKS)
