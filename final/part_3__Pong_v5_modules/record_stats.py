import logging,csv,os
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%H:%M:%S')
def record_stats(ret,ent,upd,logfile='train_stats.csv'):
    e=os.path.isfile(logfile)
    with open(logfile,'a',newline='') as f:
        w=csv.writer(f)
        if not e:w.writerow(['update','avg_return','entropy'])
        w.writerow([upd,ret,ent])