from scipy.stats import norm
from data_imports import *
from build_co_optimization import *






class Scenario:


    def __init__(self):

        # ----------- Scenarios for sensitivity analysis ------------------#

        self.storage_size = 600  # in [MWh]
        self.eff = 0.92  # [0,1] usually 75% to 90%
        self.s_max = 200  # AC interconnection limit in [MW]
        self.storage_power = 100  # in [MW]
        self.solar_plant_size = 100  # in [MW]
        self.eta = 95


    def Deterministic(self, grid_limit, ml):

        # ---------------------------- Imports ----------------------------#
        E_price, solar_cf, ASM_price = InputsSolarMarket()

        if ml==True:
            ASM_price_markov, ASM_price_pers, ASM_price_rr, ASM_price_arx = InputsMarketML()

            # ---------------------------- Solver ----------------------------#

            # Create model type and select solver
            model = pyo.ConcreteModel()
            opt = pyo.SolverFactory('glpk')

            # 1 # Perfect forecast scenario - deterministic
            # Instantiate class
            m = HybridDeterministic(model, E_price, solar_cf, ASM_price_arx) #change here the ASM ML prices

            # ---------------------- Build model, launch optimization ------------------------#
            hybrid = m.build(model, self.storage_size, self.eff, self.s_max, self.storage_power, self.solar_plant_size,
                             grid_limit)

            # Create a 'dual' suffix component on the instance so the solver plugin will know which suffixes to collect
            hybrid.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

            results = opt.solve(hybrid, tee=True)
            # results = opt.solve(hybrid, tee=False)

            # ------------------------ Post-Processing ------------------------#
            # Deterministic
            filename = 'ResultsDet.xlsx'
            print('hey')

            Max_Revenue = ResultsAnalysisML(hybrid, filename, ASM_price, E_price)


        else:
            # ---------------------------- Solver ----------------------------#

            # Create model type and select solver
            model = pyo.ConcreteModel()
            opt = pyo.SolverFactory('glpk')


            # 1 # Perfect forecast scenario - deterministic
            # Instantiate class
            m = HybridDeterministic(model, E_price, solar_cf, ASM_price)

            # ---------------------- Build model, launch optimization ------------------------#
            hybrid = m.build(model, self.storage_size, self.eff, self.s_max, self.storage_power, self.solar_plant_size, grid_limit)

            # Create a 'dual' suffix component on the instance so the solver plugin will know which suffixes to collect
            hybrid.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

            results = opt.solve(hybrid, tee=True)
            # results = opt.solve(hybrid, tee=False)

            # ------------------------ Post-Processing ------------------------#
            # Deterministic
            filename = 'ResultsDet.xlsx'
            print('ho')

            Max_Revenue = ResultsAnalysisDet(hybrid, filename)

        return Max_Revenue


    def CC_normal(self, grid_limit, hourly=False):
        # ---------------------------- Imports ----------------------------#
        E_price, solar_cf, ASM_price = InputsSolarMarket()
        solar_mean_cf, solar_std_cf, inv_cdf = InputsSolarUncertainMul(self.eta)

        inv_cdf = norm.ppf(0.95)

        # ---------------------------- Solver ----------------------------#

        # Create model type and select solver
        model = pyo.ConcreteModel()
        opt = pyo.SolverFactory('glpk')

        # 2 # Average hourly value scenario - chance contraints
        m = HybridCC_normal(model, E_price, ASM_price, solar_mean_cf, solar_std_cf, inv_cdf)


        # ---------------------- Build model, launch optimization ------------------------#
        hybrid = m.build(model, self.storage_size, self.eff, self.s_max, self.storage_power, self.solar_plant_size, grid_limit)

        # Create a 'dual' suffix component on the instance so the solver plugin will know which suffixes to collect
        hybrid.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        results = opt.solve(hybrid, tee=True)
        # results = opt.solve(hybrid, tee=False)

        # ------------------------ Post-Processing ------------------------#
        #  Chance constraints
        filename = 'ResultsCC_normal.xlsx'

        true_solar= solar_cf*self.solar_plant_size
        Max_Revenue_opt, Max_Revenue_for = ResultsAnalysisCC(hybrid, filename, true_solar)
        print('Optimal revenue optimization:', Max_Revenue_opt)

        return Max_Revenue_for

    def CC_cdf(self, grid_limit, hourly=False):
        # ---------------------------- Imports ----------------------------#
        E_price, solar_cf, ASM_price = InputsSolarMarket()
        solar_mean_cf, solar_std_cf, inv_cdf = InputsSolarUncertainMul(self.eta)


        if hourly==True:
            seasonal = True
            print(seasonal)
            inv_cdf = InputsSolarUncertainHourly(self.eta, seasonal)
            print('This is new cdf', inv_cdf)
            print(type(inv_cdf))

        # ---------------------------- Solver ----------------------------#

        # Create model type and select solver
        model = pyo.ConcreteModel()
        opt = pyo.SolverFactory('glpk')

        # 3 # Average hourly value scenario - chance contraints
        m = HybridCC_cdf(model, E_price, ASM_price, inv_cdf)

        # ---------------------- Build model, launch optimization ------------------------#
        hybrid = m.build(model, self.storage_size, self.eff, self.s_max, self.storage_power, self.solar_plant_size, grid_limit)

        # Create a 'dual' suffix component on the instance so the solver plugin will know which suffixes to collect
        hybrid.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        results = opt.solve(hybrid, tee=True)
        # results = opt.solve(hybrid, tee=False)

        # ------------------------ Post-Processing ------------------------#
        # Chance constraints
        filename = 'ResultsCC_cdf.xlsx'

        true_solar = solar_cf * self.solar_plant_size
        Max_Revenue_opt, Max_Revenue_for = ResultsAnalysisCC(hybrid, filename, true_solar)
        print('Optimal revenue optimization:', Max_Revenue_opt)

        return  Max_Revenue_for



def main():
    """
        main:
            - Imports AS ERCOT Down Regulation data ???, Solar CF
            - Runs optimization in Certain solar resource scenario
            - Results post-processing: dual variables ???, plotting, sensitivity analysis ???
           """

    s = Scenario()
    grid_limit = 0
    hourly = True #for hourly cdf in CC scenarios
    ml = True
    print("ML comparison - yes (True)/ no(False) :", ml)
    print("Grid charging - yes (1)/ no(0) :", grid_limit)

    # ---------------------- Build scenario------------------------#
    #max_rev = s.CC_cdf(grid_limit, hourly)  # change scenarios here CC_normal(grid_limit) or CC_cdf(grid_limit)
    max_rev = s.Deterministic(grid_limit, ml) #change scenarios here for ML cases
    print('Max Revenue forecasted', max_rev)



def results_plots():

    # ---------------------- Scenario results ------------------------#
    #SCENARIO 1: Deterministic
    s1_det = [29109252.49351159, 30081076.65191792]
    #no grid charging:  29109252.49351159
    #grid charging:  30081076.65191792



    # SCENARIO 2: CC normal
    s2_cc_normal = [29115840.815133967, 30081077.022033863]
    # no grid charging: 29115840.815133967
    # grid charging: 30081077.022033863


    #Constant value in the cdf
    # SCENARIO 3: CC cdf
    s3_cc_cdf_cte = [29995224.320353903, 30081077.022033863]
    # no grid charging: 29995224.320353903
    # grid charging: 30081077.022033863

    # Hourly value in the cdf
    # SCENARIO 4: CC cdf
    s4_cc_cdf_hourly = [28767588.028343957,30081077.022033863]
    # no grid charging:28767588.028343957
    # grid charging:30081077.022033863


    # Hourly value in the cdf
    # SCENARIO 4: CC cdf
    s4_cc_cdf_season= [28767146.496943984, 30081077.022033863]
    # no grid charging: 28767146.496943984
    # grid charging:30081077.022033863


    #ML Scenarios
    #Markov
    ml_markov = [28058283.240790006, 28695215.58278002]
    # no grid charging:28058283.240790006
    # grid charging: 28695215.58278002



    # Persistence
    ml_pers = [28773713.053150073, 29720109.37477001]
    # no grid charging:28773713.053150073
    # grid charging: 29720109.37477001

    # RF
    ml_rf = [28588295.201400027,29339417.82105008]
    # no grid charging: 28588295.201400027
    # grid charging: 29338383.694100082

    # ARX
    ml_arx = [28397013.68621, 28932103.15623003]
    # no grid charging:28397013.68621
    # grid charging: 28932103.15623003



    # ---------------------- Deterministic v Uncertain results ------------------------#

    # ##Create sceanrio comparison - bar chart
    # scenarios = [s1_det[0], s2_cc_normal[0], s3_cc_cdf_cte[0], s1_det[1], s2_cc_normal[1], s3_cc_cdf_cte[1]]
    # data = np.divide(scenarios, 1000000)
    #
    # # Make a fake dataset
    # bars = ('2018 Onsite', 'CC Onsite + norm', 'CC Onsite + cte cdf', '2018 w.Grid', 'CC Grid + norm', 'CC Grid + cte cdf')
    # y_pos = np.arange(len(bars))
    # col= ['darkcyan', 'darkturquoise', 'powderblue', 'olivedrab', 'yellowgreen', 'darkseagreen']
    #
    # plt.ylabel('Revenue [million USD $]', size=11)
    # plt.ylim([25, 31])
    # plt.bar(y_pos, data, color=col)
    # plt.xticks(y_pos, bars)
    # plt.show()

    # ---------------------- Deterministic v ML results ------------------------#

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)


    x_values = ('Only onsite charging', 'Grid charging allowed')
    x_pos = [0,0.5]

    ##Create sceanrio comparison - bar chart
    col = ['darkcyan', 'darkturquoise', 'powderblue', 'olivedrab', 'yellowgreen', 'darkseagreen']

    ax.scatter(x_pos, np.divide(ml_pers, 1000000), alpha=0.8, c=col[0], edgecolors='none', s=80,
               label='Persistence model')
    ax.scatter(x_pos, np.divide(ml_rf, 1000000), alpha=0.8, c=col[1], edgecolors='none', s=80,
               label='Random forest model')
    ax.scatter(x_pos, np.divide(ml_markov, 1000000), alpha=0.8, c=col[2], edgecolors='none', s=80, label='Markov chains')

    ax.scatter(x_pos, np.divide(ml_arx, 1000000), alpha=0.8, c=col[5], edgecolors='none', s=80,
               label='ARX model')

    ax.scatter(x_pos, np.divide(s2_cc_normal, 1000000), alpha=0.8, marker='^', c='sandybrown', edgecolors='none', s=80,
               label='CC normal dist.')

    ax.scatter([0,0.55], np.divide(s3_cc_cdf_cte, 1000000), alpha=0.8, marker='^', c='purple', edgecolors='none', s=80,
               label='CC hist. cte')

    ax.scatter([0.05,0.57], np.divide(s4_cc_cdf_hourly, 1000000), alpha=0.8, marker='^', c='gold', edgecolors='none', s=80,
               label='CC hist. TWD')

    ax.scatter([-0.05,0.45], np.divide(s4_cc_cdf_season, 1000000), alpha=0.8, marker='^', c='violet', edgecolors='none', s=80,
               label='CC hist. TWSD')
    ax.scatter(x_pos, np.divide(s1_det, 1000000), alpha=1, c='black', marker='+', edgecolors='none', s=130,
               label='Perfect foresight')



    plt.ylabel('Revenue [million USD $]', size=11)
    plt.xlim([-0.2, 0.8])
    plt.xticks(x_pos, x_values)
    plt.legend(bbox_to_anchor=(0,1.02,1, 0.2), loc="lower left", mode="expand", ncol=5)
    plt.show()






    #if __name__ == "__main__":
    #main()