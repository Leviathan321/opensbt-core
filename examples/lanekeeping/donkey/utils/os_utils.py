import platform
import subprocess
from examples.lanekeeping.global_log import GlobalLog

def kill_donkey_simulator() -> None:

    logg = GlobalLog("kill_donkey_simulator")

    plt = platform.system()
    
    if plt.lower() == "linux":
        keyword = "donkey"
        try:
            # Execute the pkill command with the specified keyword
            subprocess.run(['pkill', '-f', keyword], check=True)
            print(f"All processes containing '{keyword}' have been terminated.")
        except subprocess.CalledProcessError:
            print(f"No processes containing '{keyword}' found.")
    else:
        beamng_program_name = "donkey_sim"

        # windows
        cmd = "tasklist"

        ret = subprocess.check_output(cmd)
        output_str = ret.decode("utf-8")

        program_name = beamng_program_name
        if program_name in output_str:
            cmd = 'taskkill /IM "{}.exe" /F'.format(program_name)
            ret = subprocess.check_output(cmd)
            output_str = ret.decode("utf-8")
            logg.info(output_str)
        else:
            logg.warn("The program {} is not in the list of currently running programs".format(beamng_program_name))
