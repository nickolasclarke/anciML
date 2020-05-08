import pandas as pd
import pyomo.environ as pyo
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def InputsSolarMarket():

    # ------------------------ Inputs ------------------------------#

    # Day Ahead Energy Market 2018
    dataset_E = pd.read_csv('all_ercot_profiles_hourly_2018.csv')
    E_price = dataset_E.iloc[0:8762, 4].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Day Ahead Market - AS Down Regulation ERCOT hourly prices 2018
    dataset_AS = pd.read_csv('AS_price.csv')
    ASM_price = dataset_AS.iloc[70080:78840, 9].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Solar CF
    dataset_solar = pd.read_csv('all_ercot_profiles_hourly_2018.csv')  # Reading the dataset of solar gen CF
    solar_cf = dataset_solar.iloc[0:8762, 1].values  # Creating the solar generation Vector, start 1/1/2018 00:00 (CST)

    return E_price, solar_cf, ASM_price

class SolarPlusStorageCertain:

    """
        Pyomo implementation of LP hybrid, with perfect DAM foresight and certain solar resource
        """

    def __init__(self, model, E_price, solar_cf,ASM_price):

        self.model = model
        self.model.IDX = pyo.RangeSet(0, 8759)
        self.model.E_price = E_price
        self.model.ASM_price = ASM_price
        self.model.solar_cf = solar_cf

    def build(self, model, storage_size, eff, s_max, storage_power, solar_plant_size):

        # ------------------ Parameters ---------------------------#

        #Sizing based on: https://www.nrel.gov/docs/fy19osti/71714.pdf

        model.storage_size = storage_size  # in [MWh]
        model.eff = eff # [0,1] usually 75% to 90%
        model.s_max = s_max  # AC interconnection limit in [MW]
        model.storage_power = storage_power  # in [MW]
        model.solar_plant_size = solar_plant_size # in [MW]
        model.solar_gen = model.solar_plant_size * model.solar_cf # in [MW]

        # --------------- Optimization variables ------------------#

        model.negcharge = pyo.Var(model.IDX, bounds=(0.00, model.storage_power))
        model.poscharge = pyo.Var(model.IDX, bounds=(0.00, model.storage_power))
        model.energy_gen = pyo.Var(model.IDX, bounds=(-model.storage_power, model.s_max))  # Interconnection limit included in bounds
        model.soc = pyo.Var(model.IDX, bounds=(0.00, model.storage_size))  # SOC bounds include size of storage system
        model.reg = pyo.Var(model.IDX, bounds=(0.00, model.storage_power*2)) # bounds for regulation product provided
        model.E0 = 0 # Storage starts empty
        model.grid_limit = 0 ## no grid charging = 0, grid chargeing = 1
        model.reg_penalty = 0.2 ## how much energy lost by providing regulation service

        # -------------------- Objective fct. ---------------------#

        # Objective function
        def Objective_rule(m):
            expr = sum([model.E_price[t] * model.energy_gen[t]  for t in model.IDX]+ [model.ASM_price[t] * model.reg[t]  for t in model.IDX])   
            return expr

        model.Max_Revenue = pyo.Objective(rule=Objective_rule, sense=pyo.maximize)

        # ----------------- Constraints ----------------------------#

        # Grid balance equation
        def grid_balance_rule(model, t):
            return model.energy_gen[t] == model.negcharge[t] - model.poscharge[t] + model.solar_gen[t]

        model.grid_balance_const = pyo.Constraint(model.IDX, rule=grid_balance_rule)


        # Battery SOC equation
        def storage_soc_rule(model, t):
            if t == model.IDX.first():
                expr = model.soc[t] == model.E0 + model.poscharge[t] - model.eff * model.negcharge[t] / model.eff + model.reg[t] * model.reg_penalty 
            else:
                expr = model.soc[t] == model.soc[t - 1] + model.eff * model.poscharge[t] - model.negcharge[t] / model.eff + model.reg[t] * model.reg_penalty 
            return expr

        model.storage_soc_const = pyo.Constraint(model.IDX, rule=storage_soc_rule)


        # Simultaneous rule
        def limit_simultaneous_rule(model, t):
            return model.negcharge[t] + model.poscharge[t] <=  model.storage_power

        model.limit_simultaneous_const = pyo.Constraint(model.IDX, rule=limit_simultaneous_rule)

        #grid charging rule
        def only_charge_from_solar_rule(model, t):
            return model.energy_gen[t] >= min(0, -model.storage_power*model.grid_limit)
        model.only_charge_from_solar_const = pyo.Constraint(model.IDX, rule=only_charge_from_solar_rule)

        #regulation product
        def regulation_product_rule(model, t):
            return model.reg[t] <= model.storage_power + model.negcharge[t] - model.poscharge[t]
        model.regulation_product = pyo.Constraint(model.IDX, rule=regulation_product_rule)

        return model

class SolarPlusStorageUncertain(SolarPlusStorageCertain):

    """
        Pyomo implementation of SOCP ??? Ask Scott
        I tried to follow the idea of HW3 PB5, but I am not sure how to model storage component (or how to write constrain in pyomo)
        I think Appendix A of this article can help: https://arxiv.org/pdf/1906.04108.pdf
        """

    def build(self, model):
        model = super().build(model)

        # Uncertainty parameters and variables
        model.sig = pyo.Var(3, bounds=(0.00, model.storage_power))  # Should change to 0-1 ?
        model.a_av = [-50, -50, 0]
        model.cone_axis = np.matrix('[50, 0, 0; 0, 50, 0; 0, 0, 0]')

        # Add Cone uncertainty of solar generation constraint
        def solar_uncertainty(model, t):
            return  np.transpose(model.a_av) * model.sig + pyo.norm(np.transpose(model.cone_axis) * model.sig, 2) <= 0
        model.solar_uncertainty_const = pyo.Constraint(model.IDX, rule=solar_uncertainty)

        # Add limits to sigma constrain #constraints += [sig[0] <= 1, sig[0] >= 0, sig[1] <= 1, sig[1] >= 0, sig[2] == model.energy_gen]


        return model



def ResultsAnalysis(model,variable,type_v):

    # ------------------------ Optimization result ---------------------#

    #Objective value in $
    Max_Revenue = pyo.value(model.Max_Revenue)

    # ------------------------ Optimization variables ---------------------#


    # Save the schedule in python object
    solar_gen = pd.Series(model.solar_gen)
    energy_price = pd.Series(model.E_price)
    as_price = pd.Series(model.ASM_price)
    energy_gen = pd.Series([round(model.energy_gen[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])
    negcharge = pd.Series([round(model.negcharge[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])
    poscharge = pd.Series([round(model.poscharge[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])
    soc = pd.Series([round(model.soc[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])
    reg = pd.Series([round(model.reg[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])

    schedule = pd.DataFrame({'solar_gen':solar_gen,'energy_price':energy_price,'as_price':as_price,
    'solarplusstorage':energy_gen, 'storage_out':negcharge, 'storage_in':poscharge, 'storage_soc':soc, 'regulation_sell':reg})

    # #Save the schedule in an excel
    writer = pd.ExcelWriter('Results_' + str(variable) +'_'+type_v+'.xlsx')
    schedule.to_excel(writer, 'OptVariables')

    # # -------------------------- Dual variables ---------------------#

    # #Save dual variable values in python object and excel
    # i=0
    # for c in model.component_objects(pyo.Constraint, active=True):
    #     name = ['Grid balance rule', 'Storage SOC rule', 'Simultaneous rule', 'grid charging rule', 'Regulation product']
    #     dual_vb = pd.Series([round(model.dual[c[index]], 3) for index in c], index=[t for t in model.IDX])
    #     constraint = pd.DataFrame({name[i]: dual_vb})

    #     constraint.to_excel(writer, name[i])
    #     i=i+1

    writer.save()

    # Save the optimiaztion variables in csv

    # with open('results_vb.csv', 'w') as f:
    #     f.write('Hour, NegCharge, PosCharge, SOC, DispatchEnergy\n')
    #     for t in range(0,8760):
    #         f.write('%s, %s, %s, %s, %s\n' % (t, pyo.value(model.negcharge[t]), pyo.value(model.poscharge[t]), pyo.value(model.soc[t]), pyo.value(model.energy_gen[t])))


    # ------------------------- Basic Plot ---------------------#

    #OptimV_Results = pd.read_csv("results_vb.csv")
    #OptimV_Results.plot(subplots=True, figsize=(6, 6))

    return Max_Revenue



def main():
    """
        main:
            - Imports AS ERCOT Down Regulation data ???, Solar CF
            - Runs optimization in Certain solar resource scenario
            - Results post-processing: dual variables ???, plotting, sensitivity analysis ???
           """

    # ---------------------------- Imports ----------------------------#
    E_price, solar_cf, ASM_price = InputsSolarMarket()

    # ----------- Scenarios for sensitivity analysis ------------------#

    ##[Guangxuan] you can create intervals, or lists of possible values here for the different scenarios

    storage_size = 600  # in [MWh]
    eff = 0.92  # [0,1] usually 75% to 90%
    s_max = 200  # AC interconnection limit in [MW]
    storage_power = 100  # in [MW]
    solar_plant_size = 100  # in [MW]
    
    sensitivity_range=np.arange(0.8,1.2,0.2)#range of sensitivity analysis 

    # ---------------------------- Solver ----------------------------#

    # Create model
    model = pyo.ConcreteModel()
    opt = pyo.SolverFactory('glpk')

    #Instantiate class and model
    m = SolarPlusStorageCertain(model, E_price, solar_cf,ASM_price)
    hybrid = m.build(model, storage_size, eff, s_max, storage_power, solar_plant_size)

    # Create a 'dual' suffix component on the instance so the solver plugin will know which suffixes to collect
    hybrid.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    results = opt.solve(hybrid, tee=True)

    # ------------------------ Post-Processing ------------------------#

    Max_Revenue = ResultsAnalysis(hybrid,1,"base")
    print('Optimal revenue value:', Max_Revenue)
    
    #sensitivity analysis
    
    length_range=len(sensitivity_range)
    
    storage_size_range=storage_size*sensitivity_range
    eff_range = eff*sensitivity_range
    solar_plant_size_range = solar_plant_size*sensitivity_range
    s_max_range = s_max*sensitivity_range
    storage_power_range=storage_power*sensitivity_range
    
    revenue_storage_size=np.zeros(length_range)
    revenue_eff=np.zeros(length_range)
    revenue_solar_plant_size=np.zeros(length_range)
    revenue_s_max=np.zeros(length_range)
    revenue_storage_power=np.zeros(length_range)
    
    for index in range(length_range):
        
        #storage_size
        storage_size_temp=storage_size_range[index]
        a = SolarPlusStorageCertain(model, E_price, solar_cf,ASM_price)
        hybrid_a = a.build(model, storage_size_temp, eff, s_max, storage_power, solar_plant_size)
        hybrid_a.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        results_a = opt.solve(hybrid_a, tee=True)
        revenue_storage_size[index] = ResultsAnalysis(hybrid_a,storage_size_temp,"storage_energy")

        #eff
        eff_temp=eff_range[index]
        b = SolarPlusStorageCertain(model, E_price, solar_cf,ASM_price)
        hybrid_b = b.build(model, storage_size, eff_temp, s_max, storage_power, solar_plant_size)
        hybrid_b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        results_b = opt.solve(hybrid_b, tee=True)
        revenue_eff[index]= ResultsAnalysis(hybrid_b,eff_temp,"efficiency") 

        #solar_plant_size
        solar_plant_size_temp=solar_plant_size_range [index]
        c = SolarPlusStorageCertain(model, E_price, solar_cf,ASM_price)
        hybrid_c = c.build(model, storage_size, eff, s_max, storage_power, solar_plant_size_temp)
        hybrid_c.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        results_c = opt.solve(hybrid_c, tee=True)
        revenue_solar_plant_size[index] = ResultsAnalysis(hybrid_c,solar_plant_size_temp,"solar_size")
 
        #s_max
        s_max_temp=s_max_range [index]
        d = SolarPlusStorageCertain(model, E_price, solar_cf,ASM_price)
        hybrid_d = d.build(model, storage_size, eff, s_max_temp, storage_power, solar_plant_size)
        hybrid_d.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        results_s = opt.solve(hybrid_d, tee=True)
        revenue_s_max[index] = ResultsAnalysis(hybrid_d,s_max_temp,"interconnection")

        #storage_power
        storage_power_temp=storage_power_range [index]
        e = SolarPlusStorageCertain(model, E_price, solar_cf,ASM_price)
        hybrid_e = e.build(model, storage_size, eff, s_max, storage_power_temp, solar_plant_size)
        hybrid_e.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        results_e = opt.solve(hybrid_e, tee=True)
        revenue_storage_power[index] = ResultsAnalysis(hybrid_e,storage_power_temp,"storage_power")    

    #plot
    
    plt.figure(figsize=(7,4))
    
    plt.plot(sensitivity_range,revenue_storage_size/(10**6), lw = 1.5,label = 'storage duration')
    plt.plot(sensitivity_range,revenue_eff/(10**6), lw = 1.5, label = 'efficiency')
    plt.plot(sensitivity_range,revenue_solar_plant_size/(10**6), lw = 1.5, label = 'solar plant size') 
    plt.plot(sensitivity_range,revenue_s_max/(10**6), lw = 1.5,label = 'interconnection')
    plt.plot(sensitivity_range,revenue_storage_power/(10**6), lw = 1.5, label = 'storage power')
    
    
    plt.legend(loc = 0) 
    plt.axis('tight')
    plt.xlabel('Change Ratio')
    plt.ylabel('Revenue($ millions USD)')
    plt.title('sensitivity analysis')
    plt.savefig("sensitivity.png")
    plt.show()
    
    
    

if __name__ == "__main__":
    main()
    


    
        