using Plots
using Random
using Distributions
using Printf
using DelimitedFiles
using DataFrames
using CSV

# First define constants
const NBPATTERNS = 4
const NBNEUR = 200
const NBIN = 3  # Number of inputs. Input 0 is reserved for a 'go' signal that is not used here.
const NBOUT = 1  # Only 1 output neuron
const PROBACONN = 1.0  # Dense connectivity
const G = 1.5   # Early chaotic regime. Chaos ma non troppo.
const METHOD = "DELTAX"
const ETA = 0.1 # learning rate

const MODULTYPE = "DECOUPLED"
const ALPHAMODUL = 16.0 # Note that TAU = 30ms, so the real ALPHAMODUL is 16.0 * dt / TAU ~= .5
const MAXDW = 3e-4
const PROBAMODUL = .003

const RNGSEED = 0
const DEBUG = 0

const ALPHATRACE = .75
const ALPHATRACEEXC = 0.05

const SQUARING = 1;

# trial running parameters
const TRIALTIME = 1000
const STARTSTIM1 = 1
const TIMESTIM1 = 200
const STARTSTIM2 = 400
const TIMESTIM2 = 200
const EVALTIME = 200
const PHASE = "LEARNING"

const dt = 1.0
const tau = 30.0
const dtdivtau = dt / tau

const bernConnection = Multinomial(1, [PROBACONN,1. - PROBACONN])
# const bernPerturbation = Multinomial(1, [PROBAMODUL,1. - PROBAMODUL])

logTrials = [5,105,205,905]

# Here we define the inputs to be fed to the network,
# and the expected response, for each condition.
# For the sequential-XOR problem (NBPATTERNS to 4, TRIALTIME and eval. time as appropriate):
# We encode the input patterns as matrices with NBIN rows and TRIALTIME columns,
# which we fill with the appropriate input values at every time step and for each input channel
patterns = zeros(NBPATTERNS,NBIN,TRIALTIME) # initialize inputs as 3D array

# AA
patterns[1,2,STARTSTIM1 : STARTSTIM1 + TIMESTIM1] .= 1.0
patterns[1,2,STARTSTIM2 : STARTSTIM2 + TIMESTIM2] .= 1.0
# AB
patterns[2,2,STARTSTIM1 : STARTSTIM1 + TIMESTIM1] .= 1.0
patterns[2,3,STARTSTIM2 : STARTSTIM2 + TIMESTIM2] .= 1.0
# BA
patterns[3,3,STARTSTIM1 : STARTSTIM1 + TIMESTIM1] .= 1.0
patterns[3,2,STARTSTIM2 : STARTSTIM2 + TIMESTIM2] .= 1.0
# BB
patterns[4,3,STARTSTIM1 : STARTSTIM1 + TIMESTIM1] .= 1.0
patterns[4,3,STARTSTIM2 : STARTSTIM2 + TIMESTIM2] .= 1.0

# Target responses - what the network ought to produce
# (note that only the last EVALTIME timesteps are actually relevant - see below)
tgtresps = zeros(NBPATTERNS,TRIALTIME)
tgtresps[1,:] .= -.98 # same
tgtresps[2,:] .= .98 # diff
tgtresps[3,:] .= .98 # diff
tgtresps[4,:] .= -.98 # same

# initialize weight and gradient matrices
function runTrials(nbtrials)
    df = DataFrame(trial = Int[], trialtype = Int[],target = Float64[],response = Float64[])
    # Input weights are uniformly chosen between -1 and 1,
    # except possibly for output cell (not even necessary).
    # No plasticity for input weights.
    win = rand(Float64,(NBNEUR,NBIN))
    win[:,1] .= 0 # Go (input 0) weight is 0

    # Randomize recurrent weight matrix, according to the
    # Sompolinsky method (Gaussian(0,1), divided by sqrt(ProbaConn*N)
    #  and multiplied by G - see definition of randJ() below).
    C = reshape(rand(bernConnection,NBNEUR * NBNEUR)[1,:],(NBNEUR,NBNEUR))
    W = G * randn(NBNEUR,NBNEUR) ./ sqrt(PROBACONN * NBNEUR)
    J = C .* W

    # save weight distn
    open("./data/J0.txt", "w") do io
        writedlm(io, J)
    end

    # If in the TESTING mode, read in the weights
    if PHASE == "TESTING"
        print("Reading trained weights [TODO]")
    end

    # initializing more zeroed vectors
    meanerrs = zeros(nbtrials)
    r_trace = zeros(NBNEUR)
    total_exc = zeros(NBNEUR)
    x_trace = zeros(NBNEUR)
    rs = zeros(NBNEUR,TRIALTIME)
    err = zeros(TRIALTIME)
    meanerrtrace = zeros(NBPATTERNS)

    let meanerr_tracker = [[] for i = 1:NBPATTERNS]

    # OK, let's start the experiment:
    # This is where the fun begins - Anakin Skywalker
    for numtrial = 1:nbtrials

        trialtype = numtrial % NBPATTERNS + 1 # seq through types

        # We use native-C array hebbmat for fast computations
        # within the loop, then transfer it back to Eigen
        # matrix hebb for plasticity computations
        # instead of doing this, only use hebb
        hebb = zeros(NBNEUR,NBNEUR)
        r = zeros(NBNEUR)
        rs = zeros(NBNEUR,TRIALTIME)
        input = zeros(NBNEUR)
        modulmarker = zeros(NBNEUR)

        # Initialization of network activity x with moderate
        # random noise. Decreases performance a bit, but more
        # realistic.

        x = rand(NBNEUR) .* .1
        # set a subset of neuron as biases
        x[[2,10,11]] .= [1.,1.,-1.]

        r = broadcast(tanh,x)

        # step through timesteps fo the trial
        for numiter = 1:TRIALTIME
            input =  patterns[trialtype,:,numiter]
            rprev = r
            lateral_input =  J * r # recurrent activity

            total_exc = lateral_input + win * input

            # leave off here 5:16 6/3/2020
            # now apply the perturbation
            if (numiter > 3 && PHASE == "LEARNING")
                modul = perturbation()
                total_exc .+= modul
            end

            # Compute network activations
            x .+= dtdivtau .* (-x .+ total_exc)
            x[[2,10,11]] .= [1.,1.,-1.] # Biases

            # Actual responses = tanh(activations)
            r = broadcast(tanh,x) # [tanh(nn) for nn in x ]
            rs[:,numiter] .= r # log activations

            # Leave off here 12:12 AM 6/5/2020
            # Okay, now for the actual plasticity....
            # this is where the fun really begins

            # First, compute the fluctuations of neural activity
            # (detrending / high-pass filtering)
            delta_x =  x - x_trace
            x_trace = ALPHATRACEEXC .* x_trace .+ (1.0 - ALPHATRACEEXC) .* x

            if (numiter > 3 && PHASE == "LEARNING")
                updateElg!(hebb,rprev,delta_x,modul)
            end
        end

        # Trial complete!
        #  Compute error for this trial
        err = rs[1,:] - tgtresps[trialtype,:] # seems neuron 1 is output neuron here
        # Error is only computed over the response period, i.e. the last EVALTIME ms.
        err[1:TRIALTIME - EVALTIME] .= 0
        meanerr = sum(broadcast(abs,err)) / EVALTIME

        # Compute the actual weight change, based on eligibility trace and the relative
        # error for this trial:

        if ((PHASE == "LEARNING") && (numtrial > 100))
            # Note that the weight change is the summed Hebbian increments, multiplied by the
            # relative error, AND the mean of recent errors for this trial type - this last
            # multiplication may help to stabilize learning.
            # i.e. go down the gradient of erorr
            dJ = updateWeights!(J,hebb,meanerr,meanerrtrace[trialtype])
        else
            dJ = zeros(NBNEUR,NBNEUR)
        end

        # change meanerrtrace by delta rule
        meanerrtrace[trialtype] = ALPHATRACE * meanerrtrace[trialtype] + (1.0 - ALPHATRACE) * meanerr
        meanerrs[numtrial] = meanerr

        # now just log and display stuff
        append!(meanerr_tracker[trialtype],meanerr)
        push!(df,(numtrial,trialtype,tgtresps[trialtype,end],rs[1,end]))
        if PHASE == "LEARNING"
            if numtrial % 50 == 0
                print("-----\n")
                Printf.@printf("Trial %i of type: %i complete with meanerr: %f\n",numtrial,trialtype,meanerr)
                Printf.@printf("Final Timestep Response: % f, expected response : %f\n",rs[1,end],tgtresps[trialtype,end])
                Printf.@printf("r[1:4]: [%f,%f,%f,%f]\n",r[1],r[2],r[3],r[4])
                Printf.@printf("J[1,1:4]: [%f,%f,%f,%f]\n",J[1,1],J[1,2],J[1,3],J[1,4])
                Printf.@printf("dJ[1,1:4]: [%f,%f,%f,%f]\n",dJ[1,1],dJ[1,2],dJ[1,3],dJ[1,4])
                Printf.@printf("hebb[1,1:4]: [%f,%f,%f,%f]\n",hebb[1,1],hebb[1,2],hebb[1,3],hebb[1,4])
                print("-----\n")
            end
            if numtrial in logTrials
                # make these trial-type specific
                open(Printf.@sprintf("./data/rs_trial%i.txt",numtrial), "w") do io
                    writedlm(io, rs)
                end
            end
        end

        # Training variables to save: J, win, meanerrs
        # Testing variables to save: print
    end

    fname = "./data/behavior_df.csv"
    CSV.write(fname, df)
    plot(meanerr_tracker,title = "Mean error sep by Trialtype", xlabel = "Time", ylabel = "Mean error",label = ["Trialtype 1" "Trialtype 2" "Trialtype 3" "Trialtype 4"], lw = 3)
end
end

function updateWeights!(J,hebb,meanerr,meanerrtrace_tt)
    dJ = (hebb .* (-ETA * meanerrtrace_tt * (meanerr - meanerrtrace_tt)))'
    J .+= broadcast(x -> if x <= 0 max(x,-MAXDW) else min(x,MAXDW) end, dJ)
    # some number of J are going to NaN!
    if size(findall(x -> isnan(x),J),1) > 0
        print(string(isnan(meanerr),"\t"))
        print(string(isnan(meanerrtrace_tt),"\t"))
        print(string(size(findall(x -> isnan(x),hebb)),"\t"))
        print(string(size(findall(x -> isnan(x),dJ),1)),"\n")
    end
    return dJ
end

function perturbation()
    modul = zeros(NBNEUR)
    if MODULTYPE == "UNIFORM"
        # Apply a modulation to the entire network with probability PROBAMODUL -
        # Not used for these simulations.
        if rand() < PROBAMODUL
            modul .= ALPHAMODUL .* (2.0 .* rand(NBNEUR) - 1.0)
        end
    elseif MODULTYPE == "DECOUPLED"
        for i = 1:NBNEUR
            if rand() < PROBAMODUL
                modul[i] = ALPHAMODUL * (2.0 * rand() - 1.)
            end
        end
    else
        error("Which Modulation Type?")
    end
    return modul
end

function updateElg!(hebb,rprev,delta_x,modul)
    if METHOD == "DELTAX"
        for n1 = 1:NBNEUR
            for n2= 1:NBNEUR
                incr = rprev[n1] * delta_x[n2];
                hebb[n1,n2] += incr ^ 3
            end
        end
    elseif (METHOD == "NODEPERT")
        # Node-perturbation.
        # The Hebbian increment is the inputs times the
        # perturbation itself. Node-perturbation method, similar to
        # Fiete & Seung 2006. Much faster because you only compute
        # the Hebbian increments in the few timesteps at which a
        # perturbation actually occurs.
        perturb_ix = findall(x -> x > 0,modul)
        hebb[:,perturb_ix] .+= rprev * modul[perturb_ix]'
    else error("Which method??")
    end
end

# function elgIncrement!(hebb,rprev,delta_x,modul)
#     # Compute the Hebbian increment to be added to the eligibility trace
#     # (i.e. potential weight change) for this time step, based on inputs
#     # and fluctuations of neural activity
#         if (METHOD == "DELTAX")
#             # Method from the paper. Slow, but biologically plausible (-ish).
#             # The Hebbian increment at every timestep is the inputs
#             # (i.e. rprev) times the (cubed) fluctuations in activity for
#             # each neuron.
#             # Multiply cubed (ie supralinear) prev activations by highpass
#             incr = @. (rprev * delta_x')^3
#             hebb .+= incr
#
#         # not supported currently
#         elseif (METHOD == "NODEPERT")
#             # Node-perturbation.
#             # The Hebbian increment is the inputs times the
#             # perturbation itself. Node-perturbation method, similar to
#             # Fiete & Seung 2006. Much faster because you only compute
#             # the Hebbian increments in the few timesteps at which a
#             # perturbation actually occurs.
#             perturb_ix = findall(x -> x > 0,modul)
#             hebb[:,perturb_ix] .+= rprev * modul[perturb_ix]'
#
#         else error("Which method??")
#     end
# end

# function perturbationDot()
#     modul = zeros(NBNEUR)
#     if MODULTYPE == "UNIFORM"
#         # Apply a modulation to the entire network with probability PROBAMODUL -
#         # Not used for these simulations.
#         if ((rand() < PROBAMODUL))
#             modul .= 2.0 .* rand(NBNEUR) - 1.0
#         end
#     elseif MODULTYPE == "DECOUPLED"
#         perturb_ix = findall(x -> x == 1, rand(bernPerturbation,NBNEUR)[1,:])
#             modul[perturb_ix] .= 2.0 .* rand(length(perturb_ix)) .- 1.0
#     else
#         error("Which Modulation Type?")
#     end
#     return modul
# end
