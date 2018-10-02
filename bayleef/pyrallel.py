



def pyrallel(cmd, logdir='/dev/null', njobs=100, timeout="200%", *args):
    if njobs = -1:
        njobs = 0

    sub=['parallel',
         '--progress',
         '--joblog {}'.format(lofdir)
         '--jobs {}'.format(njobs),
         '--timeout={}'.format(timeout),
         '--out={}'.format(log),
         ''

         ]
    sub.append('--wrap="{}"'.format(wrap.strip()))
    # print(" ".join(sub))
    process = sps.Popen(" ".join(sub), shell=True, stdout=sps.PIPE)
    stdout = process.communicate()[0].decode("utf-8")
    return(stdout)
