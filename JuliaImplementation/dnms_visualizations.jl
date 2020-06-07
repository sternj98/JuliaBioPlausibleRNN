using Plots
using DelimitedFiles
using CSV
using DataFrames
using Statistics
using StatsPlots

# plot initial weight distribution
function plotDist()
    fname = "../dnms/J0.csv"

    J0DistJulia = readdlm("./data/J0.txt", '\t', Float64, '\n')
    J0DistCPP = readdlm(fname,',', Float64, '\n',skipblanks=true)

    histogram([vcat(J0DistJulia...) vcat(J0DistCPP...)],label = ["Julia" "CPP"])
    title!("Initial Weight Distribution Between Julia and CPP")
end

# plot neuron responses to capture learning effect on activity
function plotResponses()
    trial5 = readdlm("rs_trial5.txt", '\t', Float64, '\n')
    trial105 = readdlm("rs_trial105.txt", '\t', Float64, '\n')
    trial205 = readdlm("rs_trial205.txt", '\t', Float64, '\n')
    trial905 = readdlm("rs_trial905.txt", '\t', Float64, '\n')

    p1 = plot(trial5[1:6,:]',title = "Trial 5 Responses", xlabel = "Time", ylabel = "Activity", lw = 3)
    p2 = plot(trial105[1:6,:]',title = "Trial 105 Responses", xlabel = "Time", ylabel = "Activity", lw = 3)
    p3 = plot(trial105[1:6,:]',title = "Trial 205 Responses", xlabel = "Time", ylabel = "Activity", lw = 3)
    p4 = plot(trial905[1:6,:]',title = "Trial 905 Responses", xlabel = "Time", ylabel = "Activity", lw = 3)
    plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
end

# plot behavior over time
function plotBehavior()
    df = DataFrame(CSV.File("./data/behavior_df.csv"))
    df.trialtype = broadcast(to_stim,df.trialtype)

    p1 = @df df[(df.trial .> 0) .& (df.trial .< 500),:] boxplot(:trialtype, :response, title = "Trials 0-500")
    p2 = @df df[(df.trial .> 500) .& (df.trial .< 1000),:] boxplot(:trialtype, :response, title = "Trials 500-1000")
    p3 = @df df[(df.trial .> 1000) .& (df.trial .< 1500),:] boxplot(:trialtype, :response, title = "Trials 1000-1500")
    p4 = @df df[(df.trial .> 1500) .& (df.trial .< 2000),:] boxplot(:trialtype, :response, title = "Trials 1500-2000")

    plot(p1,p2,p3,p4,layout = (1,4),legend = false,suptitle = "DNMS Behavior Over Training Time")

end

function to_stim(x)
   if x == 1
       return "AA"
   elseif x== 2
       return "AB"
   elseif x == 3
       return "BA"
   else
       return "BB"
   end
end
