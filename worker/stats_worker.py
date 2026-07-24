import schedule
import time

schedule.every(10).minutes.do(update_global_stats)

while True:
    schedule.run_pending()
    time.sleep(1)