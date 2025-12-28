import cobra
import optlang
import os
import pandas as pd
from copy import deepcopy
from joblib import Parallel, delayed
import sys
import warnings
from cobra.flux_analysis import flux_variability_analysis
from optlang.interface import OPTIMAL
import numpy as np
from optlang.symbolics import add
from egc import resolve_egc
import random
from cobra.flux_analysis import flux_variability_analysis

warnings.filterwarnings("ignore")


def phenotypes():
    solvers = [match.split("_interface")[0] for match in dir(optlang) if "_interface" in match]
    if "cplex" not in solvers:
        raise RuntimeError("cplex not found.")

    # read in arguments
    if len(sys.argv) > 2:
        raise RuntimeError('at most one parameter is supplied.')
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    else:
        if os.path.exists('input_parameters.txt'):
            input_file = 'input_parameters.txt'
        else:
            raise RuntimeError(
                'input file not specified and the default input_parameters.txt cannot be found as well.'
            )

    # read input parameters
    paras = read(input_file)

    # load reaction pools
    universe = cobra.io.read_sbml_model(paras['REACTION_POOL'])
    universe.solver = 'cplex'

    # add reactions
    # compute fermentation flux
    print('——————————————————————————————————* phenotypes *——————————————————————————————————')
    df_output = None
    retLst = Parallel(n_jobs=int(paras['NUM_CPUS']))(
        delayed(predict_fermentation)(gem_file, universe, paras) for gem_file in paras['GEMs'].split(';')
    )
    if df_output is None:
        df_output = deepcopy(pd.concat(retLst))
    else:
        df_output = pd.concat([df_output, pd.concat(retLst)])

    # write intermediate results to file
    output_file: str = "%s/%s" % (paras['OUTPUT_DIRECTORY'], paras['OUTPUT_FILENAME'])
    df_output.to_csv(output_file, index=False)
    print('phenotypes successfully done!')


def predict_fermentation(gem_file, universe, paras):
    print('predicting fermentation: %s...' % gem_file)
    # read model and add missing reactions
    model_no_gapfill, model_w_gapfill, num_rxns_added, rids_added = add_gapfilled_reactions(gem_file, universe, paras)

    # run flux balance analysis for model with and without gap filling
    fva_no_gapfill = flux_balance_analysis(model_no_gapfill, paras)
    fva_no_gapfill.columns = [c + '__no_gapfill' if c != 'reaction' else 'reaction' for c in fva_no_gapfill.columns]
    fva_w_gapfill = flux_balance_analysis(model_w_gapfill, paras)
    fva_w_gapfill.columns = [c + '__w_gapfill' if c != 'reaction' else 'reaction' for c in fva_w_gapfill.columns]
    fva = pd.merge(fva_no_gapfill, fva_w_gapfill, left_on='reaction', right_on='reaction', how='inner')

    # expand fva
    fva['gem_file'] = gem_file.rstrip('.xml')
    fva['random_rxns'] = paras['ADD_RANDOM_RXNS']
    fva['num_rxns_to_add'] = paras['NUM_GAPFILLED_RXNS_TO_ADD']
    fva['num_rxns_added'] = num_rxns_added
    fva['rxn_ids_added'] = ';'.join(rids_added)

    # find reactions that lead to phenotypic changes from 0 to 1
    key_rxns = []
    for ex, phe1, phe2 in zip(fva['reaction'], fva['phenotype__no_gapfill'], fva['phenotype__w_gapfill']):
        if phe1 == 0 and phe2 == 1:
            model_w_gapfill2 = deepcopy(model_w_gapfill)
            model_w_gapfill2.reactions.get_by_id(ex).lower_bound = 0.1  # some nontrivial small number
            indicator_vars = []
            for rid in rids_added:
                var = model_w_gapfill2.problem.Variable('indicator_var_' + rid, lb=0, ub=1, type='binary')
                indicator_vars.append(var)
                con1 = model_w_gapfill2.problem.Constraint(
                    (model_w_gapfill2.reactions.get_by_id(rid).flux_expression + 1000.0 * var).expand(),
                    name='constr1' + rid,
                    lb=0
                )
                con2 = model_w_gapfill2.problem.Constraint(
                    (model_w_gapfill2.reactions.get_by_id(rid).flux_expression - 1000.0 * var).expand(),
                    name='constr2' + rid,
                    ub=0
                )
                model_w_gapfill2.add_cons_vars([var, con1, con2])
                model_w_gapfill2.solver.update()

            model_w_gapfill2.objective = add(*indicator_vars)
            model_w_gapfill2.objective.direction = "min"
            model_w_gapfill2.solver.update()

            skip_this_prediction = False
            for tol in [1e-9, 1e-8, 1e-7, 1e-6]:
                model_w_gapfill2.solver.problem.parameters.mip.tolerances.integrality.set(tol)
                try:
                    sol = model_w_gapfill2.optimize()
                    break
                except:
                    if tol == 1e-6:
                        skip_this_prediction = True
            if skip_this_prediction:
                key_rxns.append(np.NaN)
            else:
                if sol.objective_value >= 1:
                    key_rxns2 = []
                    for var in model_w_gapfill2.variables:
                        rid = var.name.lstrip('indicator_var_')
                        if var.primal == 1:
                            print(ex, var.name)
                        if var.primal == 1 and rid in rids_added:
                            key_rxns2.append(rid)
                    key_rxns.append(';'.join(key_rxns2))
                    print('predict_fermentation: %s can be gapfilled by %s' % (ex, key_rxns[-1]))
                else:
                    key_rxns.append(np.NaN)
        else:
            key_rxns.append(np.NaN)
    fva['essential_rxns'] = key_rxns

    return fva


# This function adds exchange reactions for targeted fermentation products
def add_ex_reactions(model, universe, target_ex_rxns):
    for rid in target_ex_rxns:
        if rid not in model.reactions:
            #print('add_ex_rxns: adding %s...' % rid)
            reaction = cobra.Reaction(rid)
            reaction.name = 'R_%s' % rid
            reaction.lower_bound = -1000.0
            reaction.upper_bound = 1000.0
            met_id = rid.lstrip('EX_')
            if met_id in model.metabolites:
                met = model.metabolites.get_by_id(met_id)
            else:
                met = universe.metabolites.get_by_id(met_id)
            reaction.add_metabolites({met: -1.0})
            model.add_reactions([reaction])
    return None


# constrain nutrient uptake
def constrain_media(model, df_cm, namespace, rxn_suffix):
    for ex in model.exchanges:
        ex_cpd = ex.id.lstrip('EX_')
        ex_cpd = ex_cpd.rstrip(rxn_suffix)
        if ex_cpd in list(df_cm[namespace]):
            ex.lower_bound = -np.abs(df_cm.loc[df_cm[namespace] == ex_cpd, 'flux'].values[0])
            ex.upper_bound = 1000.0
        else:
            ex.lower_bound = 0.0
            ex.upper_bound = 1000.0
    return None


# test if adding reactions will increase growth rate
def test_growth_inflation(model, rxns_to_add):
    max_growth = model.slim_optimize()
    model2 = deepcopy(model)
    model2.add_reactions(rxns_to_add)
    max_growth2 = model2.slim_optimize()
    if np.abs(max_growth2 - max_growth) <= 1e-6:
        return 0
    else:
        return 1


# add reactions predicted from deep learning model or randomly selected from reaction pools
def add_gapfilled_reactions(gem_file, universe, paras):
    # read parameters
    namespace = paras['NAMESPACE']
    batch_size = int(paras['BATCH_SIZE'])
    num_rxns_to_fill = int(paras['NUM_GAPFILLED_RXNS_TO_ADD'])
    add_random_rxns = int(paras['ADD_RANDOM_RXNS'])
    target_ex_rxns = paras['TARGET_EX_RXNS']
    ex_suffix = paras['EX_SUFFIX']

    # read GEM into cobrapy
    model = cobra.io.read_sbml_model("%s/%s.xml" % (paras['GEM_DIRECTORY'], gem_file))
    model.solver = 'cplex'
    print('add_gapfilled_reaction: model %s loaded' % gem_file)

    # read culture medium
    df_cm = pd.read_csv(paras['CULTURE_MEDIUM'])
    if namespace not in list(df_cm.columns):
        raise RuntimeError('cannot find %s as a column of media file.' % namespace)

    # add exchange reactions
    add_ex_reactions(model, universe, target_ex_rxns)

    # constrain boundary reactions by culture media
    constrain_media(model, df_cm, namespace, ex_suffix)

    # skip the model if it does not grow
    max_growth = model.slim_optimize()
    if max_growth == 0.0:
        raise RuntimeError("model %s cannot grow in current medium." % (gem_file.rstrip('.xml')))
    else:
        print('add_gapfilled_reaction: max growth rate = %2.2f' % max_growth)

    #  model before reaction added
    model_no_gapfill = deepcopy(model)
    model_w_gapfill = deepcopy(model)

    if add_random_rxns:
        # randomize reactions in universal pools
        candidate_reactions = list(set([r.id for r in universe.reactions if r.id not in model_w_gapfill.reactions]))
        random.shuffle(candidate_reactions)
    else:
        # read deep learning model predicted reactions with scores
        score_file = "%s/%s.csv" % (paras['GAPFILLED_RXNS_DIRECTORY'], gem_file)
        df_gapfill = pd.read_csv(score_file, index_col=0)
        # keep gapfilled reactions that are not contained in the model but included in the reaction pools
        df_gapfill = df_gapfill.loc[
            [rid for rid in df_gapfill.index if rid not in model_w_gapfill.reactions and rid in universe.reactions]
        ]
        df_gapfill = df_gapfill[df_gapfill['predicted_scores']>=float(paras['MIN_PREDICTED_SCORES'])].sort_values('similarity_scores')
        candidate_reactions = list(df_gapfill.index)
    print('add_gapfilled_reaction: find %d candidate reactions' % len(candidate_reactions))

    # if Resolve_EGC is True, add reaction one by one; otherwise add all together
    rids_added = []
    num_rxns_before = len(model_w_gapfill.reactions)
    counter = 0
    counter2 = 0
    max_counter = np.min([num_rxns_to_fill, len(candidate_reactions)])

    if int(paras['RESOLVE_EGC']):
        while counter < max_counter:

            # determine which reactions to add for this batch
            rxns_to_add = []
            for rid in candidate_reactions[counter2:]:
                if len(rxns_to_add) >= batch_size or counter + len(rxns_to_add) >= max_counter:
                    break

                assert not rid.startswith('EX_')
                rxn = universe.reactions.get_by_id(rid)

                # anaerobic condition; do not add reactions involving oxygen
                if int(paras['ANAEROBIC']):
                    if namespace == "bigg":
                        if 'o2_c' in [met.id for met in rxn.reactants] or 'o2_c' in [met.id for met in rxn.products]:
                            counter2 += 1
                            continue
                    elif namespace == "modelseed":
                        if 'cpd00007_c0' in [met.id for met in rxn.reactants] or 'cpd00007_c0' in [met.id for met in
                                                                                                   rxn.products]:
                            counter2 += 1
                            continue
                rxns_to_add.append(rxn)
                counter2 += 1

            # add reactions in batch and test growth inflation
            is_inflated = test_growth_inflation(model_w_gapfill, rxns_to_add)
            if is_inflated:
                # growth inflation found, add reaction one by one, and resolve potential egc
                for rxn in rxns_to_add:
                    assert rxn.id not in model_w_gapfill.reactions
                    is_inflated = test_growth_inflation(model_w_gapfill, [rxn])
                    if is_inflated:
                        model_w_gapfill2 = deepcopy(model_w_gapfill)
                        model_w_gapfill2.add_reactions([rxn])
                        egc_resolved = resolve_egc(model_w_gapfill2, rxn.id, namespace)
                        if egc_resolved and model_w_gapfill2.slim_optimize() < 2.81:
                            model_w_gapfill = deepcopy(model_w_gapfill2)
                            rids_added.append(rxn.id)
                            counter += 1
                            print(
                                'add_gapfilled_reaction: added %s after resolving egc, current counter = %d'
                                % (rxn.id, counter))
                        else:
                            print(
                                'add_gapfilled_reaction: failed to add %s and egc not resolved, current counter = %d'
                                % (rxn.id, counter))
                    else:
                        model_w_gapfill.add_reactions([rxn])
                        rids_added.append(rxn.id)
                        counter += 1
                        print(
                            'add_gapfilled_reaction: added %s, no growth inflation, current counter = %d'
                            % (rxn.id, counter))
            else:
                for rxn in rxns_to_add:
                    assert rxn.id not in model_w_gapfill.reactions
                model_w_gapfill.add_reactions(rxns_to_add)
                rids_added.extend([rxn.id for rxn in rxns_to_add])
                counter += len(rxns_to_add)
                print('add_gapfilled_reaction: added %d reactions, no growth inflation, current counter = %d'
                      % (len(rxns_to_add), counter))
    else:
        max_counter = np.min(paras['NUM_GAPFILLED_RXNS_TO_ADD'], len(candidate_reactions))
        rxns_to_add = candidate_reactions[0:max_counter]
        model_w_gapfill.add_reactions([rxns_to_add])

    # calculate number of reactions added
    num_rxns_after = len(model_w_gapfill.reactions)
    num_rxns_added = num_rxns_after - num_rxns_before
    assert num_rxns_added <= max_counter
    print("model %s: successfully added %d reactions." % (gem_file.replace('.xml', ''), num_rxns_added))

    return model_no_gapfill, model_w_gapfill, num_rxns_added, rids_added


def flux_balance_analysis(model, paras):
    # run flux balance analysis
    # try different linear programming method the default algorithm fails
    fba_solution = model.optimize()
    if model.solver.status != OPTIMAL:
        is_optimal = False
        for lp_method in ["primal", "dual", "network", "barrier", "sifting", "concurrent"]:
            model.solver.configuration.lp_method = lp_method
            fba_solution = model.optimize()
            if model.solver.status == OPTIMAL:
                is_optimal = True
                break
        assert is_optimal
    assert fba_solution.objective_value > 0.0

    # run parsimonious flux balance analysis
    # try different linear programming method the default algorithm fails


    pfba_solution = cobra.flux_analysis.pfba(model)

    # modify flux bounds to minimize input fluxes that do not contribute to growth
    # For flux > 0, set its lower bound to 0
    # For flux <= 0, set its lower bound to the flux value
    for ex in model.exchanges:
        assert ex.id.startswith('EX_')
        if pfba_solution.fluxes[ex.id] >= 0.0:
            ex.lower_bound = 0.0
        else:
            ex.lower_bound = pfba_solution.fluxes[ex.id]

    # run flux variability analysis
    fva = flux_variability_analysis(
        model,
        paras['TARGET_EX_RXNS'],
        fraction_of_optimum=0.999999,
        loopless=True
    )
    fva.index.name = 'reaction'
    fva = fva.reset_index()
    fva['biomass'] = fba_solution.objective_value
    fva['normalized_maximum'] = fva['maximum'] / fva['biomass']
    fva['phenotype'] = (fva['normalized_maximum'] >= float(paras['FLUX_CUTOFF'])).astype(int)
    return fva


def read(input_file):
    df = pd.read_csv(input_file, header=None)
    df.columns = ["Field", "Value"]

    # The following fields are mandatory
    mandate_fields: list[str] = [
        'CULTURE_MEDIUM',  # csv file with NAMESPACE and flux
        'REACTION_POOL',   # a universal genome scale model
        'GEM_DIRECTORY',  # directory of genome-scale metabolic models
        'GAPFILLED_RXNS_DIRECTORY',  # reactions predicted to be missing
        'NUM_GAPFILLED_RXNS_TO_ADD',  # number of reactions to add
        'ADD_RANDOM_RXNS', # 1 if adding random reactions
        'SUBSTRATE_EX_RXNS',  # csv file with NAMESPACE
    ]
    for f in mandate_fields:
        if f not in list(df.Field):
            raise RuntimeError("Field %s is mandatory." % f)

    # The following fields are optional
    optional_fields = {
        'MIN_PREDICTED_SCORES': 0.9995, # candidate reactions w/ predicted scores below this cutoff are excluded
        'NUM_CPUS': 1,  # number of cpus to use. use -1 if using all cpus
        'EX_SUFFIX': "_e",  # exchange reaction suffix
        'RESOLVE_EGC': True,  # whether resolve energy-generating cycle
        'OUTPUT_DIRECTORY': "./",  # output file directory
        'OUTPUT_FILENAME': 'suggested_gaps.csv',  # output file name
        'FLUX_CUTOFF': 1e-5,  # cutoff flux for
        'ANAEROBIC': True,  # anaerobic fermentation?
        'BATCH_SIZE': 10,  # number of reactions added in a batch
        'NAMESPACE': 'bigg'  # currently we only support bigg and modelseed
    }

    # Keep only fields that are recognizable
    df = df[df.Field.isin(mandate_fields + list(optional_fields.keys()))]

    # Convert dataframe to dictionary
    paras = {f: v for f, v in zip(df.Field, df.Value)}

    # Assign default values for optional fields
    for f in optional_fields:
        if f not in paras:
            paras[f] = optional_fields[f]

    # make sure culture medium file exists
    if not os.path.exists(paras['CULTURE_MEDIUM']):
        raise RuntimeError('cannot find culture medium file.')

    # NAMESPACE only support 'bigg' and 'modelseed'
    if paras['NAMESPACE'] not in ['bigg', 'modelseed']:
        raise RuntimeError('unrecognized namespace %s.' % (paras['NAMESPACE']))

    # Find overlaps between GEM_DIRECTORY and GAPFILLED_RXNS_DIRECTORY
    filenames_in_GEM_DIRECTORY = [f.rstrip('.xml') for f in os.listdir(paras['GEM_DIRECTORY']) if f.endswith('.xml')]
    filenames_in_GAPFILLED_RXNS_DIRECTORY = [
        f.rstrip('.csv') for f in os.listdir(paras["GAPFILLED_RXNS_DIRECTORY"]) if f.endswith('.csv')
    ]
    overlapped_gems = list(set(filenames_in_GEM_DIRECTORY).intersection(set(filenames_in_GAPFILLED_RXNS_DIRECTORY)))
    if len(overlapped_gems) == 0:
        raise RuntimeError("cannot find gapfilled reactions for any genome-scale model.")
    else:
        # print("found %d matched GEMs and gapfilled reactions." % len(overlapped_gems))
        paras['GEMs'] = ';'.join(overlapped_gems)

    # Read target exchange reactions
    df_ex = pd.read_csv(paras['SUBSTRATE_EX_RXNS'], index_col=0)
    paras['TARGET_EX_RXNS'] = ['EX_' + cpd + paras['EX_SUFFIX'] for cpd in df_ex[paras['NAMESPACE']] if
                               str(cpd) != 'nan']

    return paras