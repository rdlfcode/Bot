import warnings
import model_tools as mt

warnings.filterwarnings("ignore")

settings = mt.load_settings()

def main():
    for p, name in [(False, "auto"), (True, "auto_pct")]:
        settings["study"]["pct"] = p
        settings["study"]["name"] = name

        mt.start_study(settings)

if __name__ == '__main__':
    main()