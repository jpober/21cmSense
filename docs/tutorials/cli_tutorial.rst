CLI Tutorial
============

In this tutorial, we'll look at how to use the CLI interface and config files to produce
a sensitivity estimate for the Phase 2 MWA array.

We will assume throughout that you are working in a directory containing the following
files::

    $ ls
    mwa_phase2_compact_antpos.txt  observation_mwa.yml  observatory_mwa.yml  ps_EoS_z9.24.txt  sensitivity_mwa.yml


These are all the configuration and data files we will need to perform the sensitivity
estimate.

Antenna Positions
~~~~~~~~~~~~~~~~~
In general you do not need a file of antenna positions -- you could alternatively specify
a function to generate them (eg. see the default example
`observatory config <https://github.com/steven-murray/21cmSense/blob/master/example_configs/observatory_hera.yml>`_).
Typically, however, you will have a given list of positions. These can be in the format
of a ``.npy`` array, a pickle file, or a simple ASCII file. In any case, the array must
be of shape ``(Nant, 3)``, where the three columns are East, North and Elevation, all
in units of metres (actually, if its a pickle file, the array could be an
`astropy Quantity <https://docs.astropy.org/en/stable/units/quantity.html#creating-quantity-instances>`_,
and specify its own units of length).
Of course, East-North-Up (ENU) are specified relative to some assumed array centre location.

The last ten lines of our MWA tile positions look like::

    $ tail mwa_phase2_compact_antpos.txt
    -2.971999999999999975e+00 1.220580000000000069e+02 3.763199999999999932e+02
    -1.698900000000000077e+01 1.220699999999999932e+02 3.764599999999999795e+02
    3.995999999999999996e+00 1.099509999999999934e+02 3.762200000000000273e+02
    -9.980000000000000426e+00 1.099440000000000026e+02 3.763299999999999841e+02
    -2.399399999999999977e+01 1.099369999999999976e+02 3.764900000000000091e+02
    5.301899999999999835e+01 1.220520000000000067e+02 3.758100000000000023e+02
    -1.698699999999999832e+01 9.781000000000000227e+01 3.763600000000000136e+02
    4.602199999999999847e+01 1.099489999999999981e+02 3.758100000000000023e+02
    3.202400000000000091e+01 1.099440000000000026e+02 3.759599999999999795e+02
    1.802499999999999858e+01 1.099590000000000032e+02 3.760699999999999932e+02

Observatory
~~~~~~~~~~~
The observatory configuration specifies attributes of your observatory, like the
antenna positions and the beam. All possible parameters are described
`here <>https://21cmsense.readthedocs.io/en/latest/reference/_autosummary/observatory/py21cmsense.observatory.Observatory.html#py21cmsense.observatory.Observatory>`_
Our observatory configuration looks like::

    $ cat observatory_mwa.yml
    antpos: mwa_phase2_compact_antpos.txt
    beam:
      class: GaussianBeam
      frequency: 155    # By default, in MHz
      dish_size: 4.0    # By default, in m
    latitude: -0.4660608447416767  # In radians
    Trcv: 100000

Notice that ``antpos`` just points to our positions file. Its location is assumed to be
relative to the location of the configuration file, unless it is an absolute path.

The beam is specified by a dictionary. The "class" tells it what kind of beam to use
(at this point, only the ``GaussianBeam`` is provided), while the frequency is specified
in MHz and the ``dish_size`` in metres.

In ``21cmSense``, all calculations are performed at a single frequency -- expected to be
the central frequency of the observation. It is expected that the bandwidth (to be
discussed later) is small enough around this frequency that the cosmic signal is not
evolving significantly over the band.

The ``dish_size`` sets a number of physical limits in the calculation. It sets the
size of the beam, which in turn sets the UV-plane resolution, and the duration of
a coherent LST-bin (to be discussed further below).

The observatory also requires a ``latitude``. This is required to determine how sky
rotation affects the modes that are observed by a given baseline over the course of a
night. We do not need a longitude without loss of generality.

Finally, ``Trcv`` sets the receiver temperature, in mK. Typically, this is not highly
influential, since the sky temperature is the dominant source of thermal noise.


Observation
~~~~~~~~~~~
Now that we've specified the attributes of the observatory, we can go on to specifying
attributes of our particular set of *observations*. All possible options are described
`here <https://21cmsense.readthedocs.io/en/latest/reference/_autosummary/observation/py21cmsense.observation.Observation.html#py21cmsense.observation.Observation>`_.
Here's what the config looks like::

    $ cat observation_mwa.yml
    observatory: observatory_mwa.yml
    integration_time: 60.0
    hours_per_day: 7
    n_days: 180
    n_channels: 100
    bandwidth: 10
    coherent: true
    bl_max: 100.0

The first thing to note here is that we're specifying the observatory, that we looked
at in the previous section. In fact, the observatory could have been fully specified
directly in this file, by making it a dictionary (with all the ``key: value`` pairs from
our ``observatory_mwa.yml`` file).

The next three parameters all have to do with how much time we are observing for.
There is a good reason they are split into three parts. The ``integration_time`` specifies
the number of seconds that the telescope integrates observations to yield a single
snapshot. All data within this time frame is averaged coherently (per baseline)
whether that's a good idea or not. In addition, observations falling within a single
UV cell and within an LST-bin are averaged together coherently. By default, an LST-bin
is considered to be the length of time it takes for a point on the sky to travel through
the FWHM of the beam. This is an approximation. You can set the length of the LST-bin
directly in the ``observation.yml`` by setting ``observation_duration`` (in minutes).
The interpretation is that within this time-frame, the sky has not changed significantly,
and that if a baseline stays in the same UV cell (remember, its UV co-ordinate changes
with time), it should be averaged coherently. The ``hours_per_day`` then effectively
gives the number of such LST-bins that are observed during the night. Different LST-bins
are averaged *incoherently*, even in the same UV cell. Finally, ``n_days`` gives the
total number of days for which these kinds of observations are averaged. Each day,
the same sky appears again, and these are assumed to be able to be added coherently.

Thus, the integration time for a given UV-cell *from a single baseline* is approximately

    .. math:: n_{\rm days} * \sqrt{\frac{\rm hours}{\rm day} \frac{1}{t_{\rm LST}} t_{\rm LST}.

In detail, this is modified slightly by rotation of the sky within an LST-bin, and how
finely that duration is sampled (i.e. ``integration_time``), but these are second-order
effects, and are baseline-dependent.

The number of channels and bandwidth purely affect the range of parallel $k$-modes
probed. We note again that the bandwidth here is not meant to be the entire bandwidth
of the instrument (hence it is not included in the ``Observatory``), rather it is the
bandwidth over which the cosmic signal is relatively stationary. Smaller bandwidths
lead to fewer low-$k$ modes observed, while smaller number of channels lead to fewer
high-$k$ modes. This can affect overall sensitivity, but typically not dramatically
(and not for a particular $k$-mode).

The ``coherent`` parameter specifies whether different baselines, if they fall
into the same UV cell in the same LST-bin, should be averaged coherently. This can have
quite an impact for certain layouts. Note that baselines that are considered *redundant*,
i.e. they have the same vector to within some user-specified tolerance, are always
averaged coherently.

Finally, the ``bl_max`` parameter specifies the maximum baseline length to include in the
analysis, in metres. We reduce this to 100 since longer baselines are more prone to systematics,
and do not add a great deal of sensitivity.

Gridding Baselines
~~~~~~~~~~~~~~~~~~
Algorithmically, the first thing to do is to grid the baselines onto the UV plane.
You do not have to do this manually, but it can be useful to do so, in order to create
an intermediate product that can be investigated and re-used in further calculations.

Let's do this::

    $ sense grid-baselines observation_mwa.yml
    finding redundancies: 100%|███████████████████████████████████████████████████████| 127/127 [00:00<00:00, 408.55ants/s]
    computing UVWs: 100%|█████████████████████████████████████████████████████████████| 118/118 [00:03<00:00, 38.14times/s]
    gridding baselines: 100%|████████████████████████████████████████████████| 2586/2586 [00:00<00:00, 10307.26baselines/s]
    There are 2586 baseline types
    Saving array file as ./drift_blmin0_blmax100_0.155GHz_arrayfile.pkl

As we can see, the code first finds baseline redundancies, up to the default tolerance.
Doing this mostly acts to improve performance in the following baseline gridding,
especially for highly redundant arrays.

Following this, the code grids the baselines. Essentially, it determines the UV-coordinate
of each baseline for each integration time within an LST (the centre of each LST bin has
the array phased to zenith, and it tracks around this point throughout the bin). It then
adds the number of redundant baselines in that group to that particular UV cell.

The important output information here is the array file, which we will have to use in
our sensitivity analysis. This file is in fact a pickled version of the entire
``Observation`` class, and can be loaded into a python interpreter. Essentially, it is
just the UV grid.

Sensitivity
~~~~~~~~~~~
The final configuration file required is ``sensitivity_mwa.yml``. Let's look at this::

    $ cat sensitivity_mwa.yml
    observation: observation_hera.yml
    horizon_buffer: 0.1
    p21: ps_EoS_z9.24.txt

Here the ``observation`` is of course the previously-specified file.
Again, we could have specified the observation directly in this file, but it helps to
separate it in order to run the gridding separately.

The ``horizon_buffer`` specifies a region of kparallel which gets thrown out due to assumed
high level of foregrounds, if ``foreground_model`` is ``moderate`` (which is the default).
This is *in addition* to the horizon line. For small baselines, this effectively sets
a "bar" below which all $k_{||}$ are thrown out. Its units are ``h/Mpc``.

Finally, ``p21`` defines a fiducial EoR power spectrum model used to determine the cosmic
variance (which is added in quadrature to the thermal variance). Note that cosmic variance
doesn't average down over time and baselines, but it does average down incoherently over
spherical modes (i.e. ``21cmSense`` assumes isotropy of the cosmic signal).

``p21`` is not a parameter that is passable to ``Sensitivity``, which instead takes a
vector of ``k`` and ``delta``. When reading from a YAML configuration, it takes a ``p21``,
which specifies a *file* containing this information. By default, output power spectra
from 21cmFAST are of the correct format to be read here, though one can pass a ``.npz``
or pickle file. To obtain the power spectrum file that we use, we use the excellent
`EoS project <http://homepage.sns.it/mesinger/EOS.html>`_. In particular, we use the
``z=9.24`` power spectrum using the fiducial faint-galaxies model, since this is the
closest redshift to our observation at 155 MHz::

    $ wget https://drive.google.com/open?id=0BzlDUW4CoPOGVE5tUXdnckF0UjA -o ps_EoS_z9.24.txt

.. warning:: Note that if this parameter is not set, ``21cmSense`` uses a default ``21cmFAST`` power
             spectrum at ``z=9.5``, which may not reflect your observation!!

Now we run the sensitivity analysis::

    $ sense calc-sense sensitivity_mwa.yml --array-file drift_blmin0_blmax100_0.155GHz_arrayfile.pkl
    Getting Thermal Variance
    calculating 2D sensitivity: 100%|███████████████████████████████████████████████| 977/977 [01:24<00:00, 12.00uv-bins/s]
    averaging to 1D: 100%|████████████████████████████████████████████████████████| 239/239 [00:06<00:00, 34.89kpar bins/s]
    Getting Sample Variance
    averaging to 1D: 100%|████████████████████████████████████████████████████████| 239/239 [00:06<00:00, 34.16kpar bins/s]
    Getting Combined Variance
    averaging to 1D: 100%|████████████████████████████████████████████████████████| 239/239 [00:06<00:00, 34.41kpar bins/s]
    Significance of detection:  0.25866956282935566


This command also outputs a file ``moderate_155.000 MHz.npz``, which contains the
standard deviation of the dimensionless power spectrum.
The output file also includes the 1D k values corresponding
to the sensitivity arrays.
By default, a simple plot is made of the 1D PS uncertainty, and is written to the file
``moderate_155.000 MHz.png``. A prefix can be prepended to these filenames by using the
``--prefix`` option, and the plotting can be turned off by setting ``--no-plot``.
