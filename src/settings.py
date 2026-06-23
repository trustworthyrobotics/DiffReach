CONFIG = {
    "DEBUG_LOG": False,  # set to True to enable detailed logging of TM contents
    "CHECK_DIM":2,
    "TRUNCATE_TO_AFFINE": False,
    "FP64_IN_CROWN": True,
    "BOUND_TIME_STEP": True # calculate the reach tubes on time steps or time durations
}
def update_config(config):
    for key, value in config.items():
        assert key in CONFIG, f"Unknown config key: {key}"
        CONFIG[key] = value
