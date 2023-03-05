import numpy as np
from scipy.stats import poisson, skewnorm
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy.linalg import inv


def generate_data(dim, nb_s):
    """Generates an adjacency matrix, vector of daily trade volume to final consumers, and a vector specifying the
    sector of each firm, for `dim` firms and `nb_s` sectors.

    The matrices and vectors are randomly generated. Firm are not allowed to have themselves as supplier. Entries of
    matrix A and vector C range between 0 and 100 in absolute value.

    An entry of matrix A can have three possible meanings. If A[i][j] > 0, it is the daily trade volume from supplier j
    to firm i, if A[i][j] < 0, it means firm j is a competitor of firm i (i.e., they belong to the same sector) and
    abs(A[i][j]) is the average amount of daily trade volume firm j supplies to its clients, if A[i][j] = 0 and i != j,
    it means that firm j is not a supplier of firm i and they are not competitors (either they are not in the same
    sector, or j does not have any clients aside from the final consumers).

    Parameters
    ----------
    dim : int
        The number of firms in the network.
    nb_s : int
        The number of sectors.

    Returns
    -------
    tuple
        A tuple of numpy.arrays consisting of the signed adjacency matrix, the vector of daily trade volume to final
        consumers, and the vector specifying the sector of each firm.
    """
    # TODO: according to how A is formed, two firms in the same sector are not competitors if one supplies to the other
    #  (or both do), but with how new suppliers are searched for, they actually would be competitors, since they are in
    #  in the same sector
    # TODO: when specifying the competitor relation, you should invert the indices. If A[i][j] > 0, firm i has the role
    #  of client/consumer, but if A[i][j] < 0, firm i has the role of supplier (and thus competitor to firm j), this
    #  will likely cause issues in the conceptual logic of the centrality score. Don't forget to update the doc after
    #  changing this. (has been changed in calculation here, check whether correct and update the doc)
    # TODO: the adjacency matrix is not symmetric in the negative entries, to make symmetric, you could take the average
    #  of the two entries A[i][j] = A[j][i] = (A_prev[i][j]+A_prev[i][j])/2. This is not an issue for the simulation
    #  model, but it is for the calculation of the centrality score.
    # assign sectors
    sector = random.choices(list(range(nb_s)), k=dim)

    # A[i][j] : the daily trade volume from supplier j to customer i
    A = np.round(np.random.rand(dim, dim) * 100)
    for i in range(dim):
        # no self-link
        A[i][i] = 0
        # break some links, otherwise the network is (almost) always complete
        # use a negatively skewed distribution to get a structure more similar to a scale-free network
        nb_break = int(np.round(skewnorm.rvs(a=-0.1*dim, loc=0.8*dim, scale=0.05*dim)))
        # random integer between 0 and (dim-2), you need at least one supplier and you can't be your own supplier
        # nb_break = int(np.round(np.random.rand(1) * (dim - 2)))

        # indices of links to break, exclude the self-link
        break_ind = random.sample(list(range(i)) + list(range(i + 1, dim)), nb_break)
        # set links to zero
        A[i][break_ind] = 0

    # create negative connections
    for i in range(dim):
        # firm with which there is no supplier/client relation
        ind_no_link = (A[i] == 0)
        # firms in the same sector are competitors
        ind_same_sec = (np.array(sector) == sector[i])
        # indices of firms with which there is no link yet and who are in the same sector
        ind_neg = np.array(range(dim))[ind_no_link * ind_same_sec]
        # exclude self-link
        ind_neg = [el for el in ind_neg if el != i]

        # number of clients each competitor has
        nb_clients = np.sum(np.transpose(A)[ind_neg] > 0, axis=0)
        # replace zeros by ones to avoid division by zero
        denum = [k if k > 0 else 1 for k in nb_clients]
        # the absolute value of the weight of a competitor link is the average amount of daily trade volume from the
        # competitor to their clients
        A = np.transpose(A)
        A[i][ind_neg] = np.array(-1 * np.sum(np.transpose(A)[ind_neg], axis=0) / denum)[ind_neg]
        A = np.transpose(A)

    # C[i] : daily trade volume from firm i to the final consumers
    C = np.round(np.random.rand(dim) * 100)

    return A, C, sector


def PN_score(adj, beta=None):
    """Calculates the PN centrality score for each firm within the network.

    If no value for `beta` is given, the optimal value as determined by [...] is used.

    Parameters
    ----------
    adj : numpy.array
        The signed adjacency matrix of the network.
    beta : float, optional
        The normalization factor. (default is None)

    Returns
    -------
    numpy.array
        A vector of PN centrality scores.
    """
    # TODO: add a reference to the original paper.
    # TODO: does the matrix need to be ony -1 and 1, or are the daily trade volume values allowed?
    # TODO: this is only for symmetric adjacency matrices, also implement for non-symmetric matrices?
    P = np.copy(adj)
    N = np.copy(adj)

    P[P < 0] = 0
    N[N > 0] = 0
    N = -1 * N

    A = P - 2 * N
    n = A.shape[0]

    ones = np.ones((n, 1))
    identity = np.diag(np.ones(n))

    if beta is None:
        # beta = 1 / (2 * n - 2)
        maxdegree = np.max((np.sum(A, axis=0), np.sum(A, axis=1)))
        beta = 1 / (2 * maxdegree)

    PN = inv(identity - beta * A) @ ones
    return PN


class SimulationModel:
    """
    Simulation model of a disruption on a supply chain network.

    Attributes
    ----------
    A : numpy.array
        Adjacency matrix of daily trade volume. A[i][j] is the daily trade volume from supplier j to firm i, or
        equivalently, the daily amount of product from supplier j that is used by firm i.
    C : numpy.array
        The daily trade volume from firm i to the final consumers is C[i].
    Pini : numpy.array
        The initial production capacity of each firm.
    Pini_full_util : numpy.array
        The production capacity of each firm, assuming they operate on 100% capacity utilization.
    damaged_ind : numpy.array
        The indices of firm that were damaged due to the disruption.
    defaults : dict
        Tracks which firms have defaulted and at what time this happened (index of a defaulted firm :
        index of iteration during which default occurred).
    delta : numpy.array
        Vector with the amount of damage in each firm due to the disruption.
    n : numpy.array
        Vector of target inventory for each firm, expressed in number of days of product use.
    param : dict
        Dictionary of simulation parameters: `gamma`: recovery rate of damaged firms, `tau`: number of days over which
        the inventory is restored to the target value, `sigma`: number of days without recovery and production in the
        damaged firms, `alpha`: number of days a firm tolerates a negative inventory of a supplier, before it tries to
        replace the supplier, `nb_iter`: number of iterations (days) the simulation is run.
    prod_cap : numpy.array
        Matrix that stores the actual production capacity for each firm on each day.
    sector : numpy.array
        A vector specifying to which sector a firm belongs. Firm i belongs to the sector represented by index sector[i].

    Methods
    -------
    actual_capacity(D, Pmax)
        Calculates the actual production capacity of a firm, based on the current maximum possible production capacity
        and the demand.
    ...

    """
    # TODO: complete class doc.
    def __init__(self, A, sector, C, p=0.1, damage_level=0.2, margin=0.1, k=9, gamma=0.5, tau=6, sigma=6, alpha=2,
                 u=0.8, nb_iter=100):
        """Initializes the simulation model with the given data and parameters.

        If model parameters are not specified, default values are used.

        Parameters
        ----------
        A : numpy.array
            Adjacency matrix of daily trade volume. A[i][j] is the daily trade volume from supplier j to firm i, or
            equivalently, the daily amount of product from supplier j that is used by firm i.
        sector : numpy.array
            A vector specifying to which sector a firm belongs. Firm i belongs to the sector represented by index
            sector[i]. The sector index should start at zero.
        C : numpy.array
            The daily trade volume from firm i to the final consumers is C[i].
        p : float, optional
            The percentage of firms that are damaged by the disruption. (default is 0.1)
        damage_level : float, optional
            The average amount of damage inflicted on affected firms. On average, (100*damage_level)% of the production
            capacity is damaged. (default is 0.2)
        margin : float, optional
            Specifies the margin around the given average damage level. The actual damage will lie between
            (damage_level - margin) and (damage_level + margin), truncated from below by zero and from above by one.
            (default is 0.1)
        k : float, optional
            The average target inventory of a firm, specified as number of days of product use. (default is 9)
        gamma : float, optional
            The recovery rate of damaged firms. (default is 0.5)
        tau : float, optional
            The number of days over which the inventory is restored to the target value. (default is 6)
        sigma : float, optional
            The number of days without recovery and production in the firms damaged after the disruption. (default is 6)
        alpha : float, optional
             The number of days a firm tolerates a negative inventory of a supplier, before it tries to replace the
             supplier. (default is 2)
        u : float, optional
             Each firm has on average (100*u)% capacity utilization. This is used to assign a maximum possible
             production capacity to each firm. (default is 0.8)
        nb_iter : int, optional
             The number of iterations (days) to run the simulation. (default is 100)
        """

        # store network data
        self.A = np.copy(A)
        self.sector = np.copy(sector)
        self.C = np.copy(C)

        # keep track of which firms have defaulted and at what time this happened
        # key: index of a defaulted firm, value: index of iteration during which default occurred
        self.defaults = {}

        # vector of target inventory for each firm, expressed in number of days of product use
        self.n = poisson.rvs(mu=k, size=self.dim())
        # dictionary with simulation parameters
        self.param = {"gamma": gamma, "tau": tau, "sigma": sigma, "alpha": alpha, "nb_iter": nb_iter}

        # vector with the amount of damage in each firm
        self.delta = self.disruption(p, damage_level, margin)
        # indices of firm that were damaged at time t=0
        self.damaged_ind = np.array(range(self.dim()))[self.delta > 0]

        # the initial production capacity of each firm
        self.Pini = self.init_capacity()

        # we assume the firms have about (100*u)% capacity utilization, we assign for each firm
        # a value which represents 100% capacity utilization, i.e. max production capacity, which
        # is the absolute upper bound of what they can produce.
        self.Pini_full_util = self.cap_full_util(u)

        # matrix to store the actual production capacity
        # row i is the production capacity at time i-1 for every firm, meaning row 0 is the production capacity before
        # the disruption and row 1 is the production capacity on the first day after the disruption (t=0)
        # column j is the production capacity of firm j throughout the entire run
        self.prod_cap = [list(self.init_capacity())]

    def run_simulation(self, print_iter=False):
        """Executes the simulation model.

        Parameters
        ----------
        print_iter : boolean, optional
            Specifies whether to print the iteration number at the start of every iteration. (default is False)
        """

        # the initial realized demand of each firm
        # The realized demand in each firm i for the product of each other firm j is given by D_star[i][j]. This gives
        # the demand that has actually been met.
        D_star = self.init_capacity()
        # the initial inventory
        S = self.setup_inventory()
        # the initial orders
        O = self.orders(self.target_inventory(D_star), S, D_star)

        A_tot = self.tot_consumption_sector()

        # days_neg_S[i][j] : number of days firm i's inventory of product from firm j has been negative
        days_neg_S = np.zeros((self.dim(), self.dim()))

        for i in range(self.param["nb_iter"]):
            if print_iter:
                print(f"iteration {i + 1}/{self.param['nb_iter']}")

            # max production capacity limited by amount of damage and inventory
            Pmax = self.max_capacity_tot(S, A_tot)

            # calculate the demand
            # The demand in each firm i for the product of each other firm j is given by D[i][j].
            D = self.demand(O)

            # calculate the actual production capacity
            Pact = self.actual_capacity(D, Pmax)

            # calculate the realized demand and orders
            D_star, C_star, O_star = self.realized_demand(Pact, D, S, O)
            # record the actual production capacity
            self.prod_cap.append(list(Pact))

            # update the inventory
            S = self.update_inventory(D_star, O_star, S)

            # record how many days the supply has been insufficient
            days_neg_S = self.update_days_neg_S(days_neg_S, S)

            # remove firms that have zero realized demand (i.e., defaulted)
            if np.sum(Pact <= 0) > 0:
                S = self.remove_firms(Pact, S, i)

            # replace suppliers if they have not been able to supply enough for more than `alpha` days
            change_supplier = (np.sum(days_neg_S > self.param["alpha"]) > 0)
            if change_supplier:
                S = self.replace_supplier(Pact, Pmax, S, days_neg_S)

                # update Pini, A_tot because self.A changed
                self.Pini = self.init_capacity()
                self.Pini[self.Pini == 0] = 1  # to avoid division by zero
                A_tot = self.tot_consumption_sector()

            # calculate O for the next day
            target_S = self.target_inventory(D_star)
            O = self.orders(target_S, S, D_star)

            # update the damage
            damaged_firms = (np.sum(self.delta) > 0)
            days_left_without_recovery = self.param["sigma"]
            if damaged_firms and (days_left_without_recovery == 0):
                # firms recover from some damage
                self.delta = self.update_delta(Pmax, D)
            elif days_left_without_recovery > 0:
                # decrease the nb of days without recovery
                self.param["sigma"] -= 1

    def actual_capacity(self, D, Pmax):
        # Pact[i] : the actual production of firm i on day t
        Pact = np.min((D, Pmax), axis=0)
        return Pact

    def calc_realized_demand_row(self, Pact, c, O_pre_ji, O_now_ji):
        # calculates the realized demand for firm i

        # c = C[i]
        O_rel_c = 1

        # O_pre_disr = A + (1/tau) * (target_S - S)
        # O_pre_ji = np.transpose(O_pre_disr)[i]
        # O_now_ji = np.transpose(O)[i]

        # make dict with the index as key, give C the highest index, exclude O_rel_ji == 0
        O_rel_dict = {j: O_now_ji[j] / O_pre_ji[j] for j in range(self.dim()) if O_pre_ji[j] != 0}
        O_rel_dict[self.dim()] = O_rel_c

        O_sub_ji = np.zeros(self.dim())
        O_sub_c = 0

        # step 1
        # r = Pact[i]
        r = Pact

        loop = len(O_rel_dict) != 0
        while loop:  # (step 7)

            # step 2
            O_rel_min = np.min(list(O_rel_dict.values()))

            # step 3
            ind = np.array(list(O_rel_dict.keys()))
            if self.dim() in ind:
                r_necessary = np.sum(O_rel_min * O_now_ji[ind[ind != self.dim()]]) + O_rel_min * c
            else:
                r_necessary = np.sum(O_rel_min * O_now_ji[ind])

            loop = r > r_necessary
            if loop:  # else go to step 8
                # step 4
                if self.dim() in ind:
                    O_sub_c += O_rel_min
                    O_sub_ji[ind[ind != self.dim()]] = O_sub_ji[ind[ind != self.dim()]] + O_rel_min
                else:
                    O_sub_ji[ind] = O_sub_ji[ind] + O_rel_min

                # step 5
                r -= r_necessary

                # step 6
                O_rel_dict = {key: value for key, value in O_rel_dict.items() if value != O_rel_min}

                loop = len(O_rel_dict) != 0

        # step 8
        if (np.sum(O_now_ji) + c) == 0:
            O_rea = 0
        else:
            O_rea = r / (np.sum(O_now_ji) + c)

        # step 9
        O_star_ji = O_rea * O_now_ji + O_sub_ji * O_now_ji
        C_star = O_rea * c + O_sub_c * c
        D_star = np.sum(O_star_ji) + C_star

        return D_star, C_star, O_star_ji

    def calc_zeta(self, Pmax, D):
        # nb_sup[i] : nb of suppliers firm i has
        nb_sup = np.sum(self.A != 0, axis=1)
        # nb_cus[i] : nb of customers firm i has
        nb_cus = np.sum(self.A != 0, axis=0)
        # nb_neigh[i] : nb of neighbours firm i has
        nb_neigh = nb_sup + nb_cus

        # healthy[0][i] : matrix of dimension (1,dim), boolean; if firm i is healthy: true; else: false
        # (Pmax >= Pini) || (Pmax >= D): a firm is healthy if the maximum possible production capacity is greater than
        # the capacity before the disruption, or if the maximum capacity is greater than the demand
        healthy = np.array([(Pmax >= self.Pini) + (Pmax >= D)])
        # matrix of dimension (dim,dim), each row is the healthy array
        # healthy_sup[i][j] : firm j is healthy (bool)
        healthy_sup = np.repeat(healthy, self.dim(), axis=0)
        # matrix of dimension (dim,dim), each column is the healthy array
        # healthy_cus[i][j] : firm i is healthy (bool)
        healthy_cus = np.repeat(np.transpose(healthy), self.dim(), axis=1)

        # nb_sup_healthy[i] : nb of healthy suppliers firm i has
        nb_sup_healthy = np.sum((self.A != 0) * healthy_sup, axis=1)
        # nb_cus_healthy[i] : nb of healthy customers firm i has
        nb_cus_healthy = np.sum((self.A != 0) * healthy_cus, axis=0)
        # nb_neigh_healthy[i] : nb of healthy neighbours firm i has
        nb_neigh_healthy = nb_sup_healthy + nb_cus_healthy

        # damping factor: ratio of number of healthy neighbours to total number of neighbours
        nb_neigh[nb_neigh == 0] = 1  # to avoid zero division, won't affect results
        zeta = nb_neigh_healthy / nb_neigh

        return zeta

    def cap_full_util(self, u):
        # TODO: see explanation in overleaf doc wrt using a prob distr for robustness
        # lower and upper bound of current capacity utilization
        lo_cap_util = np.max((0, u - 0.1))
        hi_cap_util = np.min((1, u + 0.1))
        # for each firm generate a value for the current capacity utilization
        firm_cap_util = np.array(lo_cap_util + np.random.rand(self.dim()) * (hi_cap_util - lo_cap_util))
        # calculate the maximum possible capacity utilization
        Pini_full_util = self.Pini / firm_cap_util
        return Pini_full_util

    def current_capacity(self):
        # Pcap[i] : the production capacity of firm i, defined as its maximum production assuming no supply shortages
        Pcap = self.Pini_full_util * (1 - self.delta)

        # there is no production in the damaged firms for the first sigma days after the disruption
        if self.param["sigma"] > 0:
            Pcap[self.damaged_ind] = 0

        return Pcap

    def demand(self, O):
        # D[i] : total demand for firm i on day t
        D = np.sum(O, axis=0) + self.C
        return D

    def dim(self):
        """
        Returns
        -------
        int
            The number of firms.
        """
        return self.A.shape[0]

    def disruption(self, p, damage_level, margin):
        # number of firms damaged
        d = int(np.round(p * self.dim()))
        # indices of damaged firms
        damaged_ind = random.sample(list(range(self.dim())), d)
        # delta [i] : proportion of the production capital of firm i is malfunctioning
        delta = np.zeros(self.dim())
        # TODO: use a different formula to determine the amount of damage?
        # generate numbers between 0 and 1 with mean 0.5
        damage = np.random.rand(d)
        # transform to numbers between (damage_level - margin) and (damage_level + margin) with mean damage_level
        damage = (damage_level - margin) + 2 * margin * damage
        # numbers < 0 are set to 0, numbers > 1 are set to 1
        damage = np.max((damage, np.zeros(d)), axis=0)
        damage = np.min((damage, np.ones(d)), axis=0)
        delta[damaged_ind] = damage
        return delta

    def get_prod_capacity(self):
        return np.copy(self.prod_cap)

    def group_failing_suppliers_by_sector(self, days_neg_S):
        # which suppliers do we want to replace
        ind_supp_repl = np.array(range(self.dim()))[days_neg_S > self.param["alpha"]]
        # sectors of these suppliers
        supp_sectors = list(set(self.sector[ind_supp_repl]))
        # group these in a dictionary, where the keys are the sectors,
        # the values a list of the suppliers in that sector
        sec_dict = {s: [supp for supp in ind_supp_repl if self.sector[supp] == s] for s in supp_sectors}
        return sec_dict

    def init_capacity(self):
        # Pini[i] : initial production of firm i in a day
        Pini = np.sum(self.A, axis=0) + self.C
        return Pini

    def max_capacity_inventory(self, S_tot, A_tot):
        # Ppro[i] : the maximal possible production capacity of firm i limited by the inventory of product of sector s
        if np.sum(A_tot == 0) > 0:
            # if there are zero values in A_tot, the true divide will result in nan values
            # replacing the zeros by ones and the corresponding zeros in S_tot by the max value
            # avoids the zero division, and will not affect the further calculations, since the minimum
            # of each row of Ppro will be taken
            ind_zero = (A_tot == 0)
            A_tot[ind_zero] = 1
            S_tot[ind_zero] = np.max(S_tot)
            Ppro = np.transpose(np.transpose(S_tot / A_tot) * self.Pini)
            A_tot[ind_zero] = 0

        else:
            Ppro = np.transpose(np.transpose(S_tot / A_tot) * self.Pini)

        return Ppro

    def max_capacity_tot(self, S, A_tot):

        # the current production capacity, taking into account damage from the disruption
        Pcap = self.current_capacity()

        S_tot = self.tot_inventory_sector(S)

        Ppro = self.max_capacity_inventory(S_tot, A_tot)

        # Pmax[i] : maximum possible production of firm i on day t is limited by the production capacity
        # (infrastructure, workforce, etc.) and production constraints due to supply shortages
        Ppro_min = np.min(Ppro, axis=1)
        Pmax = np.min((Pcap, Ppro_min), axis=0)
        Pmax = np.max((Pmax, np.zeros(Pmax.shape)), axis=0)
        return Pmax

    def nb_s(self):
        """
        Returns
        -------
        int
            The number of sectors
        """
        return len(set(self.sector))

    def orders(self, target_S, S, D_star):
        # O[i,j] : orders from firm i to firm j
        O = self.A * D_star / self.Pini + (1 / self.param["tau"]) * (target_S - S)
        O[O < 0] = 0
        O[list(self.defaults.keys())] = np.zeros(self.dim())
        return O

    def plot_rel_capacity(self, col_by_sector=True):
        x = list(range(self.param["nb_iter"] + 1))

        # get relative production capacity
        y = self.get_prod_capacity()
        init_p = (self.param["nb_iter"] + 1) * [list(y[0])]  # [list(y[0])]
        y = y / init_p

        # filter out firms that have always rel cap == 1
        tr_y = np.transpose(y)
        select_firms = np.array(range(self.dim()))
        data = tr_y[select_firms]
        firm_indices = np.array(range(self.dim()))[select_firms]

        # build plot df
        plot_df = pd.DataFrame(data.reshape(-1, 1), columns=["rel_prod_cap"])
        plot_df["x"] = x * len(firm_indices)
        ind_start = 0
        for ind in firm_indices:
            plot_df.loc[ind_start:(ind_start + self.param["nb_iter"]), "sector"] = self.sector[ind]
            plot_df.loc[ind_start:(ind_start + self.param["nb_iter"]), "firm"] = ind
            ind_start += self.param["nb_iter"] + 1

        # plot
        if col_by_sector:
            fig = px.line(plot_df, x="x", y="rel_prod_cap", color='sector', line_group="firm")

        else:
            fig = px.line(plot_df, x="x", y="rel_prod_cap", color="firm")

        fig.update_layout(
            xaxis_title="t (days)",
            yaxis_title="Relative production capacity"
        )
        fig.show()

    def plot_act_capacity(self, filter_firms=True, col_by_sector=True):
        # TODO: make more efficient (repeating code from prev plotting function)
        # TODO: issue with selecting lines by clicking on the labels, deselecting a sector does not make all lines
        #  disappear
        x = list(range(self.param["nb_iter"] + 1))
        y = self.get_prod_capacity()

        if filter_firms:
            # filter out firms that have always rel cap ==1
            init_p = (self.param["nb_iter"] + 1) * [list(self.Pini)]  # [list(y[0])]
            tr_y = np.transpose(y / init_p)
            select_firms = (np.sum(np.array(tr_y) == 1, axis=1) != tr_y.shape[1])
            y = np.transpose(np.transpose(y)[select_firms])
            firm_indices = np.array(range(self.dim()))[select_firms]
        else:
            firm_indices = np.array(range(self.dim()))

        if col_by_sector:
            data = np.transpose(y)
            fig = go.Figure()
            leg_entry_set = set()
            for i in range(self.dim()):
                if self.sector[i] in leg_entry_set:
                    showleg = False
                else:
                    showleg = True
                fig.add_trace(go.Scatter(x=x, y=data[i], mode='lines', name=f'sector {self.sector[i]}',
                                         line=dict(color=px.colors.qualitative.Plotly[self.sector[i]]),
                                         showlegend=showleg, legendrank=self.sector[i]))
                leg_entry_set.add(self.sector[i])
        else:
            df = pd.DataFrame(y, columns=["f" + str(el) for el in firm_indices])
            df["t"] = x
            long_df = pd.wide_to_long(df, "f", i="t", j="firm")
            t, f = zip(*long_df.index)
            long_df["t"] = list(t)
            long_df["firm"] = list(f)

            fig = px.line(long_df, x="t", y="f", color='firm')  # , markers=True

        if filter_firms:
            fig.update_layout(
                xaxis_title="t (days)",
                yaxis_title="Actual Production capacity",
                showlegend=True
            )
        else:
            fig.update_layout(
                xaxis_title="t (days)",
                yaxis_title="Actual Production capacity",
                showlegend=False
            )

        fig.show()

    def potential_suppliers_and_capacity(self, ind_not_i, sec, Pmax, Pact):
        # look for all firms in the same sector/competitors of the to be replaced suppliers
        potential_supp = np.array([ind for ind in ind_not_i if self.sector[ind] == sec])
        # calculate their free capacity
        Pfree = np.min((np.zeros(len(potential_supp)), Pmax[potential_supp] - Pact[potential_supp]), axis=0)
        # filter out the potential suppliers with Pfree == 0
        available = (Pfree > 0)
        potential_supp = potential_supp[available]
        Pfree = Pfree[available]
        return potential_supp, Pfree

    def realized_demand(self, Pact, D, S, O):
        # indices of firms where production capacity is equal to or greater than the demand
        ind_demand_met = (Pact >= D)
        # indices of firms where production is insufficients for the demand
        ind_demand_not_met = (Pact < D)

        # realized demand (demand met)
        D_star = np.zeros(self.dim())
        D_star[ind_demand_met] = D[ind_demand_met]

        C_star = np.zeros(self.dim())
        C_star[ind_demand_met] = self.C[ind_demand_met]

        O_star_T = np.zeros((self.dim(), self.dim()))  # transposed (O_ji)
        O_star_T[ind_demand_met] = np.transpose(O)[ind_demand_met]

        # realized demand (demand not met)
        target_S = self.target_inventory(D_star)  # TODO: use different formula?
        O_pre_disr = self.A + (1 / self.param["tau"]) * (target_S - S)

        for i in np.array(range(self.dim()))[ind_demand_not_met]:
            if Pact[i] > 0:
                D_star[i], C_star[i], O_star_T[i] = self.calc_realized_demand_row(Pact[i], self.C[i],
                                                                                  np.transpose(O_pre_disr)[i],
                                                                                  np.transpose(O)[i])
            else:
                D_star[i], C_star[i], O_star_T[i] = 0, 0, 0
        return D_star, C_star, np.transpose(O_star_T)

    def remove_firms(self, Pact, S, default_time):
        # if a firm has a realized demand of zero, remove it from the network, how to deal with the customers?
        # A entries for the to be removed firm are set to zero, the inventory of product of this firm is set to zero
        # if there was a negative supply in S, it is divided over other suppliers in the same sector. if there was
        # a positive amount of inventory, then what?
        #  done: idea, remove dependence in A of the to be removed firm on the staying firms, leave the dependence of
        #  the remaining firms on the to be removed firms. Set Pini and Pini_full_util to zero for the to be removed
        #  firms, set the inventory in possession of the to be removed firms to zero

        ind_firms_repl = np.array(range(self.dim()))[Pact <= 0]
        # filter out firms that have already defaulted and been removed
        # for the first `sigma` days, the firms hit by the disruption do not produce, these should not be removed even
        # though they have `Pact == 0`
        if self.param["sigma"] > 0:
            ind_firms_repl = [i for i in ind_firms_repl if i not in self.defaults.keys() and i not in self.damaged_ind]
        else:
            ind_firms_repl = [i for i in ind_firms_repl if i not in self.defaults.keys()]

        # set rows in A to zero (the removed firms won't receive trade volume anymore)
        self.A[ind_firms_repl] = np.zeros(self.dim())
        # set production capacity to zero
        # Pini is set to one, because if this is set to zero, you divide by zero in some of the calculations.
        # Instead, where necessary, the relevant values will be set to zero using self.defaulted
        self.Pini[ind_firms_repl] = 1
        self.Pini_full_util[ind_firms_repl] = 0

        # TODO: allow other firms to take over the inventory?
        # set the removed firms' inventory to zero
        S[ind_firms_repl] = np.zeros(self.dim())

        # TODO: leave like this? or let final consumers look for new suppliers?
        self.C[ind_firms_repl] = 0

        # add the indices to the firms that have defaulted
        for i in ind_firms_repl:
            self.defaults[i] = default_time

        # return the updated inventory matrix
        return S

    def replace_supplier(self, Pact, Pmax, S, days_neg_S):
        # indices of firms who want to replace one or more suppliers (they have had a negative amount of supply of at
        # least one supplier, for more than `alpha` days)
        ind_customers = np.array(range(self.dim()))[np.sum(days_neg_S > self.param["alpha"], axis=1) > 0]
        for i in ind_customers:
            # list of indices of firms which are not firm i
            ind_not_i = list(range(i)) + list(range(i + 1, self.dim()))
            # dictionary of the suppliers that need to be replaced, grouped by their sector
            # key: sector index, value: list of firm indices of the failing suppliers in that sector
            sec_dict = self.group_failing_suppliers_by_sector(days_neg_S[i])

            # for loop over the sectors
            for sec, supp_list in sec_dict.items():
                # indices of the potential suppliers and their free production capacity
                potential_supp, Pfree = self.potential_suppliers_and_capacity(ind_not_i, sec, Pmax, Pact)

                # sum the A_i,j between the firm and the to be replaced suppliers, this is what needs to be
                # attributed to the new suppliers
                A_repl = np.sum(self.A[i][supp_list])

                if A_repl > 0:
                    supp_sorted, Pfree_sorted, Pfree_cumsum = self.sort_potential_suppliers(i, potential_supp, Pfree)
                    # returns the first index such that Pfree_cumsum[k] >= A_repl,
                    # else returns len(Pfree_cumsum)-1, i.e., if sum(Pfree) < A_repl
                    last_needed_supp = next((k for k, val in enumerate(Pfree_cumsum) if val >= A_repl),
                                            len(Pfree_cumsum) - 1)

                    if last_needed_supp >= 0:  # there are potential suppliers available
                        # chosen suppliers
                        supp_chosen = supp_sorted[0:(last_needed_supp + 1)]
                        # TODO: check whether these calculations/reasoning are/is correct
                        #  how do we deal with the not covered A_repl? What if no other suppliers have free capacity?
                        #  maybe always divide the needed trade volume (i.e. the ordered amount) over all existing
                        #  suppliers? you can't divide it over non-preexisting suppliers, because they won't take on new
                        #  customers, but you can increase your orders to pre-existing suppliers.
                        #  --> instead of putting A_repl_not_covered back on the suppliers we wanted to replace, we
                        #  divide it over all current suppliers
                        # first leave all of A_repl_not_covered with the to be replaced suppliers
                        A_repl_not_covered = np.min((A_repl - Pfree_cumsum[last_needed_supp], 0))
                        # update the A matrix
                        self.A[i][supp_list] = A_repl_not_covered * self.A[i][supp_list] / A_repl
                        self.A[i][supp_chosen[0:last_needed_supp]] += Pfree_sorted[0:last_needed_supp]
                        volume_left = A_repl - A_repl_not_covered - Pfree_cumsum[last_needed_supp - 1]
                        self.A[i][supp_chosen[last_needed_supp]] += volume_left
                        # calculate the proportions, these are used to divide the negative supply
                        #proportions = self.A[i][supp_chosen] / (A_repl - A_repl_not_covered)

                    else:  # there are no potential suppliers available
                        A_repl_not_covered = A_repl

                    # redivide A_repl_not_covered over all current suppliers
                    total_trade_vol = np.sum(self.A[i])

                    defaulted_vol = 0
                    for j in supp_list:
                        if j in self.defaults.keys():
                            # if firm j has defaulted, the trade volume it delivered will be counted in total_trade_vol,
                            # but by setting A[i][j] to zero here, we ensure that no new trade volume will be attributed to
                            # the defaulted firm. Instead, the trade volume that would have been attributed to firm j is
                            # divided over the other firms in the below steps
                            defaulted_vol += self.A[i][j]
                            self.A[i][j] = 0

                    prop_trade_vol = self.A[i] / total_trade_vol
                    # set the trade vol from the to be replace suppliers to zero
                    self.A[i][supp_list] = 0
                    # redivide A_repl_not_covered according to the trade volume proportions
                    self.A[i] += prop_trade_vol * (A_repl_not_covered + defaulted_vol)

                    # update the S matrix: divide the negative inventory over the new suppliers according to the new
                    # trade volume proportions
                    S_rediv = np.sum(S[i][supp_list])
                    #S_rediv_not_covered = (A_repl_not_covered / A_repl) * S_rediv
                    #S_rediv_covered = (1 - A_repl_not_covered / A_repl) * S_rediv
                    #self.S[i][supp_list] = S_rediv_not_covered * self.A[i][supp_list] / A_repl
                    #self.S[i][supp_chosen] += S_rediv_covered * proportions
                    new_prop_trade_vol = self.A[i] / total_trade_vol
                    S[i] += new_prop_trade_vol * S_rediv
        return S

    def setup_inventory(self):
        # nb days of product each firm has in inventory at start
        n_start = np.round(np.random.rand(self.dim()) * self.n)
        n_start[n_start == 0] = 1  # always start with some inventory
        # S[i][j] : firm i has an inventory of the intermediate goods produced by firm j on day t
        S = np.transpose(n_start * np.transpose(self.A))
        return S

    def sort_potential_suppliers(self, i, potential_supp, Pfree):
        # sort the potential suppliers to determine the new suppliers
        # sorted by decreasing Pfree
        indices = np.array(range(len(potential_supp)))
        existing_ind = indices[self.A[i][potential_supp] > 0]
        new_ind = indices[self.A[i][potential_supp] == 0]

        existing_ind_sorted = sorted(existing_ind, key=lambda ind: Pfree[ind], reverse=True)
        new_ind_sorted = sorted(new_ind, key=lambda ind: Pfree[ind], reverse=True)
        total_ind_sorted = existing_ind_sorted + new_ind_sorted

        total_supp_sorted = potential_supp[total_ind_sorted]
        total_Pfree_sorted = Pfree[total_ind_sorted]
        Pfree_cumsum = np.cumsum(total_Pfree_sorted)

        return total_supp_sorted, total_Pfree_sorted, Pfree_cumsum

    def target_inventory(self, D_star):
        target_S = np.transpose(self.n * np.transpose(self.A)) * D_star / self.Pini
        target_S[list(self.defaults.keys())] = np.zeros(self.dim())
        return target_S

    def tot_inventory_sector(self, S):
        # S_tot[i][j] : total inventory in firm i of product of sector j
        S_tot = np.zeros((self.nb_s(), self.dim()))

        for i in range(self.nb_s()):
            S_tot[i] = np.sum(np.transpose(np.transpose(S)[np.array(self.sector) == i]), axis=1)

        return np.transpose(S_tot)

    def tot_consumption_sector(self):
        # A_tot[i][j] : total consumption in firm i of product of sector j
        A_tot = np.zeros((self.nb_s(), self.dim()))

        for i in range(self.nb_s()):
            A_tot[i] = np.sum(np.transpose(np.transpose(self.A)[np.array(self.sector) == i]), axis=1)

        return np.transpose(A_tot)

    def update_days_neg_S(self, days_neg_S, S):
        # it's not enough to know a firm has not been able to deliver enough for a certain
        # number of days, you need to know to whom they have not delivered, otherwise you
        # don't know by whom they're being replaced, i.e., you need a matrix, not a vector

        # indices of the firms who can't deliver enough supply
        ind_neg = (S < 0)

        # if a firm that has (days_neg_S > 0) now has a positive supply, set days_neg_S to zero
        ind_pos = ~ind_neg
        set_to_zero = ind_pos * (days_neg_S > 0)
        days_neg_S[set_to_zero] = 0

        # increment the number of days with negative supply
        days_neg_S[ind_neg] += 1

        return days_neg_S

    def update_delta(self, Pmax, D):
        # calculate delta for the next day
        delta = (1 - self.param["gamma"] * self.calc_zeta(Pmax, D)) * self.delta
        return delta

    def update_inventory(self, D_star, O_star, S):
        # calculate inventory for the next day
        S = S + O_star - self.A * D_star / self.Pini
        S[list(self.defaults.keys())] = np.zeros(self.dim())
        return S
