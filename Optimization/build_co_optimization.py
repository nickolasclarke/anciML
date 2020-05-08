import pandas as pd
import pyomo.environ as pyo
import numpy as np


class HybridDeterministic:

    """
        Pyomo implementation of LP hybrid, with perfect DAM foresight and certain solar resource
        """

    def __init__(self, model, E_price, solar_cf,ASM_price):

        self.model = model
        self.model.IDX = pyo.RangeSet(0, 8759)
        self.model.E_price = E_price
        self.model.ASM_price = ASM_price
        self.model.solar_cf = solar_cf

    def build(self, model, storage_size, eff, s_max, storage_power, solar_plant_size, grid_limit):

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
        model.energy_gen = pyo.Var(model.IDX, bounds=(-np.inf, np.inf))  # Interconnection limit included in bounds
        model.soc = pyo.Var(model.IDX, bounds=(0.00, model.storage_size))  # SOC bounds include size of storage system
        model.reg = pyo.Var(model.IDX, bounds=(0.00, model.storage_power*2)) # bounds for regulation product provided
        model.E0 = 0 # Storage starts empty
        model.grid_limit = grid_limit ## no grid charging = 0, grid chargeing = 1
        model.reg_penalty = 0.2 ## how much energy lost by providing regulation service

        # -------------------- Objective fct. ---------------------#

        # Objective function
        def Objective_rule(m):
            expr = sum([model.E_price[t] * model.energy_gen[t]  for t in model.IDX]+ [model.ASM_price[t] * model.reg[t]  for t in model.IDX])
            return expr

        model.Max_Revenue = pyo.Objective(rule=Objective_rule, sense=pyo.maximize)

        # ----------------- Constraints ----------------------------#

        # Export limits - unecertainty paramenters
        def grid_import_limit(model, t):
            return model.energy_gen[t] >= -model.storage_power*model.grid_limit

        model.grid_import_limit = pyo.Constraint(model.IDX, rule=grid_import_limit)

        # Export limits - unecertainty paramenters
        def grid_export_limit(model, t):
            return model.energy_gen[t] <= model.s_max

        model.grid_export_limit = pyo.Constraint(model.IDX, rule=grid_export_limit)


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


        #regulation product
        def regulation_product_rule(model, t):
            return model.reg[t] <= model.storage_power + model.negcharge[t] - model.poscharge[t]
        model.regulation_product = pyo.Constraint(model.IDX, rule=regulation_product_rule)

        return model

class HybridCC_normal():

    """
        Pyomo implementation of SOCP ??? Ask Scott"""

    def __init__(self, model, E_price ,ASM_price, solar_mean_cf, solar_std_cf, inv_cdf):

        self.model = model
        self.model.IDX = pyo.RangeSet(0, 8759)
        self.model.E_price = E_price
        self.model.ASM_price = ASM_price
        #self.model.solar_cf = solar_cf
        ###
        self.model.solar_mean_cf = solar_mean_cf
        self.model.solar_std_cf = solar_std_cf
        self.model.invd = inv_cdf


    def build(self, model, storage_size, eff, s_max, storage_power, solar_plant_size, grid_limit):

        # ------------------ Parameters ---------------------------#

        #Sizing based on: https://www.nrel.gov/docs/fy19osti/71714.pdf

        model.storage_size = storage_size  # in [MWh]
        model.eff = eff # [0,1] usually 75% to 90%
        model.s_max = s_max  # AC interconnection limit in [MW]
        model.storage_power = storage_power  # in [MW]
        model.solar_plant_size = solar_plant_size # in [MW]

        model.solar_gen = model.solar_plant_size * model.solar_mean_cf  # in [MW]

        # --------------- Optimization variables ------------------#

        model.negcharge = pyo.Var(model.IDX, bounds=(0.00, model.storage_power))
        model.poscharge = pyo.Var(model.IDX, bounds=(0.00, model.storage_power))
        model.soc = pyo.Var(model.IDX, bounds=(0.00, model.storage_size))  # SOC bounds include size of storage system
        model.reg = pyo.Var(model.IDX, bounds=(0.00, model.storage_power*2)) # bounds for regulation product provided
        model.E0 = 0 # Storage starts empty
        model.grid_limit = grid_limit ## no grid charging = 0, grid chargeing = 1
        model.reg_penalty = 0.2 ## how much energy lost by providing regulation service

        # -------------------- Objective fct. ---------------------#

        # Objective function
        def Objective_rule(m):
            expr = sum([model.E_price[t] * (model.negcharge[t] - model.poscharge[t] + model.solar_gen[t])  for t in model.IDX] + [model.ASM_price[t] * model.reg[t]  for t in model.IDX])
            return expr

        model.Max_Revenue = pyo.Objective(rule=Objective_rule, sense=pyo.maximize)

        # ----------------- Constraints ----------------------------#


        # Grid balance equation - UNCERTAIN parameters

        def grid_balance_rule_import(model, t):
            return (-model.poscharge[t] + model.negcharge[t] + model.solar_gen[t] + model.storage_power*model.grid_limit) >= \
                   model.invd * model.solar_std_cf[t] #could add plus model.storage_power if chatging from grid allowed

        model.grid_balance_rule_import = pyo.Constraint(model.IDX, rule=grid_balance_rule_import)


        # Grid balance equation - UNCERTAIN parameters

        def grid_balance_rule_export(model, t):
            return (model.poscharge[t] - model.negcharge[t] - model.solar_gen[t] + model.s_max) >= \
                   model.invd*model.solar_std_cf[t]

        model.grid_balance_rule_export = pyo.Constraint(model.IDX, rule=grid_balance_rule_export)


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


        #regulation product
        def regulation_product_rule(model, t):
            return model.reg[t] <= model.storage_power + model.negcharge[t] - model.poscharge[t]
        model.regulation_product = pyo.Constraint(model.IDX, rule=regulation_product_rule)

        return model


class HybridCC_cdf():

    """
        Pyomo implementation of SOCP ??? Ask Scott
        I tried to follow the idea of HW3 PB5, but I am not sure how to model storage component (or how to write constrain in pyomo)
        I think Appendix A of this article can help: https://arxiv.org/pdf/1906.04108.pdf
        """

    """
        Pyomo implementation of LP hybrid, with perfect DAM foresight and certain solar resource
        """

    def __init__(self, model, E_price , ASM_price, inv_cdf):

        self.model = model
        self.model.IDX = pyo.RangeSet(0, 8759)
        self.model.E_price = E_price
        self.model.ASM_price = ASM_price
        self.model.inv_cdf = inv_cdf


    def build(self, model, storage_size, eff, s_max, storage_power, solar_plant_size, grid_limit):

        # ------------------ Parameters ---------------------------#

        #Sizing based on: https://www.nrel.gov/docs/fy19osti/71714.pdf

        model.storage_size = storage_size  # in [MWh]
        model.eff = eff # [0,1] usually 75% to 90%
        model.s_max = s_max  # AC interconnection limit in [MW]
        model.storage_power = storage_power  # in [MW]
        model.solar_plant_size = solar_plant_size # in [MW]

        model.solar_gen = model.solar_plant_size * model.inv_cdf  # in [MW]

        # --------------- Optimization variables ------------------#

        model.negcharge = pyo.Var(model.IDX, bounds=(0.00, model.storage_power))
        model.poscharge = pyo.Var(model.IDX, bounds=(0.00, model.storage_power))
        model.soc = pyo.Var(model.IDX, bounds=(0.00, model.storage_size))  # SOC bounds include size of storage system
        model.reg = pyo.Var(model.IDX, bounds=(0.00, model.storage_power*2)) # bounds for regulation product provided
        model.E0 = 0 # Storage starts empty
        model.grid_limit = grid_limit ## no grid charging = 0, grid chargeing = 1
        model.reg_penalty = 0.2 ## how much energy lost by providing regulation service

        # -------------------- Objective fct. ---------------------#

        # Objective function
        def Objective_rule(m):
            expr = sum([model.E_price[t] * (model.negcharge[t] - model.poscharge[t] + model.solar_gen[t])  + model.ASM_price[t] * model.reg[t]  for t in model.IDX])
            return expr

        model.Max_Revenue = pyo.Objective(rule=Objective_rule, sense=pyo.maximize)

        # ----------------- Constraints ----------------------------#


        # Grid balance equation - UNCERTAIN parameters

        def grid_balance_rule_import(model, t):
            return -model.poscharge[t] + model.negcharge[t] + model.solar_gen[t]  >=  \
                   -model.storage_power*model.grid_limit

        model.grid_balance_rule_import = pyo.Constraint(model.IDX, rule=grid_balance_rule_import)


        # Grid balance equation - UNCERTAIN parameters

        def grid_balance_rule_export(model, t):
            return model.poscharge[t] - model.negcharge[t] - model.solar_gen[t] >= \
                   -model.s_max

        model.grid_balance_rule_export = pyo.Constraint(model.IDX, rule=grid_balance_rule_export)


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


        #regulation product
        def regulation_product_rule(model, t):
            return model.reg[t] <= model.storage_power + model.negcharge[t] - model.poscharge[t]
        model.regulation_product = pyo.Constraint(model.IDX, rule=regulation_product_rule)

        return model


def ResultsAnalysisDet(model, filename):

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

    schedule = pd.DataFrame({'energy_gen':energy_gen, 'solar_gen':solar_gen,'energy_price':energy_price,'as_price':as_price, 'storage_out':negcharge, 'storage_in':poscharge, 'storage_soc':soc, 'regulation_sell':reg})

    #Save the schedule in an excel
    writer = pd.ExcelWriter(filename)
    schedule.to_excel(writer, 'OptVariables')

    # -------------------------- Dual variables ---------------------#

    #Save dual variable values in python object and excel
    i=0
    for c in model.component_objects(pyo.Constraint, active=True):
        name = ['AC import', 'AC export', 'Storage SOC rule', 'Simultaneous rule', 'Grid Balance', 'Regulation product']
        dual_vb = pd.Series([round(model.dual[c[index]], 3) for index in c], index=[t for t in model.IDX])
        constraint = pd.DataFrame({name[i]: dual_vb})

        constraint.to_excel(writer, name[i])
        i=i+1

    writer.save()


    return Max_Revenue

def ResultsAnalysisML(model, filename, ASM_price, E_price):


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

    schedule = pd.DataFrame({'energy_gen':energy_gen, 'solar_gen':solar_gen,'energy_price':energy_price,'as_price':as_price, 'storage_out':negcharge, 'storage_in':poscharge, 'storage_soc':soc, 'regulation_sell':reg})


    # ------------------------ Comparison result ---------------------#

    # Objective value in $
    Max_Revenue = sum([E_price[t] * energy_gen[t] + ASM_price[t] * reg[t]  for t in range(0,8760)])

    return Max_Revenue


def ResultsAnalysisCC(model, filename, true_solar_gen):

    # ------------------------ Optimization result ---------------------#

    #Objective value in $
    Max_Revenue_opt = pyo.value(model.Max_Revenue)

    # ------------------------ Optimization variables ---------------------#


    # Save the schedule in python object
    solar_gen = pd.Series(model.solar_gen)
    energy_price = pd.Series(model.E_price)
    as_price = pd.Series(model.ASM_price)
    negcharge = pd.Series([round(model.negcharge[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])
    poscharge = pd.Series([round(model.poscharge[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])
    soc = pd.Series([round(model.soc[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])
    reg = pd.Series([round(model.reg[t].value, 3) for t in model.IDX], index=[t for t in model.IDX])

    schedule = pd.DataFrame({'solar_gen':solar_gen,'energy_price':energy_price,'as_price':as_price, 'storage_out':negcharge, 'storage_in':poscharge, 'storage_soc':soc, 'regulation_sell':reg})

    #Save the schedule in an excel
    writer = pd.ExcelWriter(filename)
    schedule.to_excel(writer, 'OptVariables')

    # -------------------------- Dual variables ---------------------#

    #Save dual variable values in python object and excel
    i=0
    for c in model.component_objects(pyo.Constraint, active=True):
        name = ['AC import', 'AC export', 'Storage SOC rule', 'Simultaneous rule', 'Regulation product']
        dual_vb = pd.Series([round(model.dual[c[index]], 3) for index in c], index=[t for t in model.IDX])
        constraint = pd.DataFrame({name[i]: dual_vb})

        constraint.to_excel(writer, name[i])
        i=i+1

    writer.save()

    Max_Revenue_for = sum([energy_price[t] * (negcharge[t] - poscharge[t] + true_solar_gen[t]) + as_price[t] * reg[t] for t in range(0, 8760)])

    return Max_Revenue_opt, Max_Revenue_for

