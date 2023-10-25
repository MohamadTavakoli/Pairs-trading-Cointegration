
#Finding the Optimal Pre-Set Boundaries for Pairs Trading Strategy Based on Cointegration Technique

#I used GM-WAT as my two cointegrated stocks

using DataFrames
using DataFramesMeta
using Dates
using AlphaVantage
using Plots

#Using AlohaVantage API to import data

client = AlphaVantage.GLOBAL[];
client.key = "";


#DATA PREPROCESSING
GM = time_series_daily("GM",outputsize="full");
GM_data = DataFrame(GM);
GM_data[!, :timestamp] = Dates.Date.(GM_data[!, :timestamp]);
GM_data[!, :open] = Float64.(GM_data[!, :open]);

GM_price=GM_data[:,"close"];
GM_price=DataFrame([GM_price],[:Daily_price_GM]);



WAT = time_series_daily("WAT",outputsize="full");
WAT_data = DataFrame(WAT);
WAT_data[!, :timestamp] = Dates.Date.(WAT_data[!, :timestamp]);
WAT_data[!, :open] = Float64.(WAT_data[!, :open]);

WAT_data=WAT_data[1:size(GM_price,1),1:6];
WAT_price=WAT_data[:,"close"];
WAT_price=DataFrame([WAT_price], [:Daily_price_WAT]);

#step 1 : Finding the Cointegration error

#Linear regression

using GLM
Prices_Dataframe=hcat(WAT_price,GM_price);
ols = lm(@formula(Daily_price_GM ~ Daily_price_WAT),Prices_Dataframe);

#Finding Coefficients
#Finding Cointegration Error
#PS1,t −βPS2,t = µ +εt
β=coef(ols)[2];

#Johansen method

ϵ=zeros(size(GM_price,1));
ϵ=DataFrame([ϵ], [:Cointegration_Error]);
for i in range(1,size(WAT_price,1),step=1)
    ϵ[i,1]=GM_price[i,1]-WAT_price[i,1]*β
end

#Engle-Granger method

μ=describe(ϵ)[:,"mean"][1];
ϵt=zeros(size(GM_price,1));
ϵt=DataFrame([ϵt], [:Cointegration_Error]);
for i in range(1,size(ϵ,1),step=1)
    ϵt[i,1]=ϵ[i,1]-μ
end
plot(ϵt[:,1])
#step 2: AR(1) Process on Cointegration Error
ϵlaged=ϵt[1:end-1,:];
ϵnew=ϵt[2:end,:];

ϵ_Dataframe=hcat(ϵlaged,ϵnew,makeunique=true);

ols1 = lm(@formula(Cointegration_Error_1 ~ Cointegration_Error),ϵ_Dataframe);
#Yt=𝓔+ΦYt-1
Φ=coef(ols1)[2]

𝓔=zeros(size(ϵlaged,1));
𝓔=DataFrame([𝓔], [:𝓔t]);
for i in range(1,size(ϵlaged,1),step=1)
    𝓔[i,1]=ϵnew[i,1]-ϵlaged[i,1]*Φ
end
𝓔t=zeros(size(𝓔,1));
𝓔t=DataFrame([𝓔t], [:Normalized_𝓔]);
for i in range(1,size(𝓔,1),step=1)
    𝓔t[i,1]=𝓔[i,1]-describe(𝓔)[1,"mean"]
end

#Step3: Obtain an approximation of the mean first-passage time

σ𝓔t=describe(𝓔t,:std)[1,"std"]
σ𝓔t2=σ𝓔t^2; #Variance

using LinearAlgebra

function Τ(a,b,h,u)
    n=(b-a)/h;
    n=Integer(round(n,digits=0));
    K_matrix=zeros(n+1,n+1);
    ones_matrix=ones(n+1,1);
    for i ∈ range(0,n,step=1)
        #ui=a+h*i
        for j ∈ range(0,n,step=1)
            if j==0 || j==n
                wj=1
            else
                wj=2
            end
            #uj=a+h*j
            K_matrix[i+1,j+1]=((h*wj)/(2*sqrt(2*π)*σ𝓔t))*exp(-1*(((a+(h*j))-Φ*(a+(h*i)))^2)/(2*σ𝓔t2))
        end
    end
    final_matrix=zeros(n+1,n+1);
    for i ∈ range(0,n,step=1)
        for j ∈ range(0,n,step=1)
            if i==j
                final_matrix[i+1,j+1]=1-K_matrix[i+1,j+1]
            else
                final_matrix[i+1,j+1]=(-1)*K_matrix[i+1,j+1]
            end
        end
    end
    τn_matrix=inv(final_matrix)*ones_matrix;
    return τn_matrix[Integer(round((((u-a)/h)+1),digits=0)),1]
end

#Trade durations
#TDU := T(a=0,b=∞,h,u=U)
#Inter-trade intervals
#IU = T(a=-∞,b=U,h,u=0)
#PeriodU=TDU +IU
# T/(TDU+IU)+1>E(NUT)>T/(TDU+IU)-1
#Minimum Total Profit and the Optimal Pre-set Upper-bound
b_=5*describe(ϵt,:std)[1,"std"]
b_=round(b_, digits=0);
function TDU(u)
    return Τ(0,b_,0.1,u)
end

function IU(u)
    return Τ(-b_,u,0.1,0)
end

function MTP(U)
    T=size(ϵt,1)
    return U*((T/(TDU(U)+IU(U)))-1)
end

MTP(5)
MTPs=zeros(size(range(0.25,b_,step=0.25),1));
MTPs=DataFrame([MTPs], [:MTPs]);

for i ∈ range(0.25,b_,step=0.25)
    MTPs[Integer(4*i),1]=MTP(i)
end
OptimalMTP=describe(MTPs,:max)[1,"max"]
Max_position=argmax(eachrow(MTPs));
OptimalU=Max_position/4
