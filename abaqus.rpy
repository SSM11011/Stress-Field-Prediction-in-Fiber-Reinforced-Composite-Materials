# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 6.14-2 replay file
# Internal Version: 2014_08_22-19.30.46 134497
# Run by SSM11011 on Sat May 02 08:02:00 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.12305, 1.11979), width=165.313, 
    height=111.083)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('data/abaqus_simulation.py', __main__.__dict__)
#: The model "Composite_nf100_49" has been created.
#: Simulation complete: job_nf100_49
print 'RT script done'
#: RT script done
