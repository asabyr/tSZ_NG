#!/usr/bin/env python
import pylauncher as launcher
job_files_dir="/scratch/07833/tg871330/tSZ_NG_ini_jobs/job_files_suite/30bin_3to3_goal_wiener_beam_MF/"
launcher.ClassicLauncher(job_files_dir+"30bin_3to3_goal_wiener_beam_MF_fid.tasks",cores=9,debug="job")