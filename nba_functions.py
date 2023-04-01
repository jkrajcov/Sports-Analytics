import requests
import pandas as pd
import numpy as np
from utils import make_joint, normalize, plot_contour, decorate, marginal, Pmf
from scipy.stats import gaussian_kde, norm, chi2
import scipy.stats as sts
import sys
from scipy.optimize import nnls
from sklearn import preprocessing

no_days_in_block = 28

headers  = {
    'Connection': 'keep-alive',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-token': 'true',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
    'x-nba-stats-origin': 'stats',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

def get_data(game_id):
    play_by_play_url = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_"+game_id+".json"
    response = requests.get(url=play_by_play_url, headers=headers).json()
    play_by_play = response['game']['actions']
    df = pd.DataFrame(play_by_play)
    df['gameid'] = game_id
    return df

def create_items_dict(items_df):
  items_df.reset_index(inplace=True, drop=True)
  dictionary = pd.Series(items_df["Teams"]).to_dict()
  items_dict = {v: k for k, v in dictionary.items()}
  return items_dict

def indices(row, items_dict, home_col_index, away_col_index):
    i = items_dict[row[home_col_index]]
    j = items_dict[row[away_col_index]]
    return i, j

def points(row, home_points_col_index, away_points_col_index):
    points_ij = row[home_points_col_index]
    points_ji = row[away_points_col_index]
    return points_ij, points_ji

def indices_and_points(row, items_dict, home_col_index, away_col_index, home_points_col_index, away_points_col_index):
    """Return indices i,j and points ij,ji for the given pair"""
    i, j = indices(row, items_dict, home_col_index, away_col_index)
    points_ij, points_ji = points(
        row, home_points_col_index, away_points_col_index)
    return i, j, points_ij, points_ji

def create_P_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                         home_points_col_ind, away_points_col_ind):
  n = len(teams_dict)
  P_matrix = np.zeros((n, n))
  for row in data_np:
    i, j, points_ij, points_ji = indices_and_points(row, teams_dict, home_col_ind, away_col_ind,
                                                    home_points_col_ind, away_points_col_ind)
    if i != j:
      if points_ij > 0 or points_ji >0:
          P_matrix[j][i] += 1
          P_matrix[i][j] += 1

  return P_matrix

def create_Tf_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                         home_points_col_ind, away_points_col_ind):
  n = len(teams_dict)
  t = np.zeros(n)
  f = np.zeros((n,1))
  T_matrix = np.zeros((n,n))
  for row in data_np:
    i, j, points_ij, points_ji = indices_and_points(row, teams_dict, home_col_ind, away_col_ind,
                                                    home_points_col_ind, away_points_col_ind)
    t[i] += 1
    t[j] += 1
    f[i] += points_ij
    f[j] += points_ji
  for x in range(0,len(t)):
    T_matrix[x][x] = t[x]
  return T_matrix, f

def massey_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                         home_points_col_ind, away_points_col_ind):
  n = len(teams_dict)
  p = np.zeros((n,1))
  p_temp = np.zeros((n,1))
  r = np.zeros((n,1))
  r_temp = np.zeros((n,1))
  M_matrix = np.zeros((n,n))
  M_temp = np.zeros((n,n))
  for row in data_np:
    i, j, points_ij, points_ji = indices_and_points(row, teams_dict, home_col_ind, away_col_ind,
                                                    home_points_col_ind, away_points_col_ind)
    M_matrix[i][j] -= 1
    M_matrix[j][i] -= 1
    M_matrix[i][i] += 1
    M_matrix[j][j] += 1
    p[i] += points_ij - points_ji
  #replace last row
  for i in range(0,n):
    M_temp = M_matrix.copy()
    p_temp = p.copy()
    M_temp[i,] = np.ones((n))
    p_temp[i] = 0
    r_temp = np.linalg.solve(M_temp,p_temp)
    r += r_temp
    r[i] -= r_temp[i]
  return r/(n-1)

def create_items_dict(items_df):
  items_df.reset_index(inplace=True, drop=True)
  dictionary = pd.Series(items_df["Teams"]).to_dict()
  items_dict = {v: k for k, v in dictionary.items()}
  return items_dict

def indices(row, items_dict, home_col_index, away_col_index):
    i = items_dict[row[home_col_index]]
    j = items_dict[row[away_col_index]]
    return i, j

def points(row, home_points_col_index, away_points_col_index):
    points_ij = row[home_points_col_index]
    points_ji = row[away_points_col_index]
    return points_ij, points_ji

def indices_and_points(row, items_dict, home_col_index, away_col_index, home_points_col_index, away_points_col_index):
    """Return indices i,j and points ij,ji for the given pair"""
    i, j = indices(row, items_dict, home_col_index, away_col_index)
    points_ij, points_ji = points(
        row, home_points_col_index, away_points_col_index)
    return i, j, points_ij, points_ji


def create_score_matrices(data_np, teams_dict, home_col_ind, away_col_ind, home_points_col_ind, away_points_col_ind):
  """Construct score matrix A and P"""
  epsilon = sys.float_info.epsilon
  n = len(teams_dict)
  A = np.zeros((n, n))
  P = np.full((n, n), epsilon)
  for row in data_np:
      i, j, points_ij, points_ji = indices_and_points(
          row, teams_dict, home_col_ind, away_col_ind,
          home_points_col_ind, away_points_col_ind)
      A[i][j] += points_ji
      A[j][i] += points_ij
  for i in range(n):
      for j in range(n):
          P[i][j] += A[i][j]
  #log_numpy_matrix(A, 'od A',2)
  return A, P 

def computation_phase(A, P, tol):
    """Compute offense, defense vectors and overall ratings."""
    size = len(A)
    e = np.ones(size)
    defense = e
    error = 1
    iter = 0
    while error > tol:
        oldobar = defense
        offense = (P.conj().transpose()) @ (1 / defense)
        defense = (P) @ (1 / offense)
        error = float(
            np.linalg.norm(oldobar * (1 / defense) - e, 1))
        iter += 1
    rating = offense / defense
    return rating, offense, defense

def create_voting_matrix(data_np, teams_dict, voting_method, home_col_ind, away_col_ind, 
                         home_points_col_ind, away_points_col_ind):
  n = len(teams_dict)
  voting_array = np.zeros((n, n))
  for row in data_np:
    i, j, points_ij, points_ji = indices_and_points(row, teams_dict, home_col_ind, away_col_ind,
                                                    home_points_col_ind, away_points_col_ind)
    
    if voting_method == 'VotingWithLosses':
      if points_ij > points_ji:
          voting_array[j][i] += 1
      elif points_ij < points_ji:
          voting_array[i][j] += 1
      else:
          voting_array[i][j] += 0.5
          voting_array[j][i] += 0.5
    elif voting_method == 'WinnersAndLosersVotePoint':
      voting_array[i][j] += points_ji
      voting_array[j][i] += points_ij
    elif voting_method == 'NetScoreDiff':
      voting_array[i][j] += points_ij - points_ji
      voting_array[j][i] += points_ji - points_ij
    elif voting_method == 'LosersVotePointDiff':
      sump = points_ij - points_ji
      if sump < 0:
          voting_array[i][j] += (-sump)
      else:
          voting_array[j][i] += sump
  return voting_array

def build_stochastic(voting_array):
  ## Build Stochastic Matrix
  n = len(voting_array)
  stochastic_matrix = np.zeros((n, n))
  for ii in range(n):
    tmp = sum(voting_array[ii])
    for i in range(n):
      if tmp == 0:
          stochastic_matrix[ii, i] = 1.0 / n
      else:
          stochastic_matrix[ii, i] = voting_array[ii, i] / tmp
  return stochastic_matrix

def compute(stochastic_matrix, b):
  n = len(stochastic_matrix)
  #stochastic_matrix_asch = (b * stochastic_matrix + ((1 - b) / n) * np.ones((n, n)))
  #stochastic_matrix_asch = b*stochastic_matrix + Beta*np.ones((n,n))*offense.T + Beta*np.ones((n,n))*defense.T
  eigenvalues, eigenvectors = np.linalg.eig(
      stochastic_matrix_asch.T)
  # Find index of eigenvalue = 1
  idx = np.argmin(np.abs(eigenvalues - 1))
  w = np.real(eigenvectors[:, idx]).T
  # Normalize eigenvector to get a probability distribution
  pi_steady = w / np.sum(w)
  rating = pi_steady
  return stochastic_matrix_asch, pi_steady, rating

def gem(voting_array,alphas,per_vectors):
  teams = len(voting_array)
  e = np.ones((teams,1))
  nalphas = len(alphas)
  npv = len(per_vectors[0])*len(per_vectors)

  St = build_stochastic(voting_array)

  G = alphas[0]*St
  if npv <= teams:                             #if feature vectors
    for i in range(1,nalphas):
      G = G + alphas[i]*e*per_vectors[:,i-1]
  else:                                       #if feature matrices
    for i in range(0,int(npv/teams)):
      S = build_stochastic(per_vectors[i])
      G = G + alphas[i+1] * S

  eigenvalues, eigenvectors = np.linalg.eig(G.T)
  idx = np.argmin(np.abs(eigenvalues - 1))
  w = np.real(eigenvectors[:, idx]).T
  # Normalize eigenvector to get a probability distribution
  pi_steady = w / np.sum(w)
  return pi_steady

def gem_matrix(voting_array,alphas,per_vectors):
  teams = len(voting_array)
  e = np.ones((teams,1))
  nalphas = len(alphas)
  npv = len(per_vectors[0])*len(per_vectors)

  St = build_stochastic(voting_array)

  G = alphas[0]*St
                                      #if feature matrices
  for i in range(0,int(npv/teams)):
    S = build_stochastic(per_vectors[i])
    G = G + alphas[i+1] * S

  eigenvalues, eigenvectors = np.linalg.eig(G.T)
  idx = np.argmin(np.abs(eigenvalues - 1))
  w = np.real(eigenvectors[:, idx]).T
  # Normalize eigenvector to get a probability distribution
  pi_steady = w / np.sum(w)
  return pi_steady

def create_vector(matrix):
  teams = len(matrix)
  w = np.zeros(teams*(teams-1), dtype=float)
  increment = 0

  for i in range(0,teams):
    for j in range(0,teams):
      if i != j:
        w[increment] = matrix[i][j]
        increment += 1
  return w

def AlphasLS(Total, no_teams):
  statcount = len(Total)
  teamcount = no_teams

  T = [[0 for i in range(teamcount)] for j in range(statcount)]

  #ELIMINATE THE DIAGONAL ZEROS IN EACH OF THE AGGREGATED MATRICES
  for i in range(0,statcount):
    for j in range(0,teamcount):
      T[i][j] = Total[i][0:j] + Total[i][(j+1):teamcount]

  #TURN EACH STATISTIC MATRIX INTO A VECTOR, THAT IS THE jth COLUMN IN THE MATRIX C
  C = np.zeros((teamcount*(teamcount-1),statcount), dtype=float) 

  for j in range(0,statcount):
    C[:, j] = [item for sublist in T[j] for item in sublist] 

  b = C[:,0]  
  C = np.delete(C, 0, 1)

  Cbar = C[~np.all(C == 0, axis=1)]
  bbar = b[~np.all(C == 0, axis=1)]

  alphas, resid_squared = nnls(Cbar, bbar)

  # CONSTRAINED LEAST SQUARES COMPUTATION OF alphas ELIMINATE ZERO ROWS
  #Matlab code not using

  alphas=(1/sum(alphas))*alphas
  return alphas, resid_squared

def LeeSeungNMF(V,k):           #this code doesn't quite work
  [n,m] = V.shape
  W = abs(random.standard_normal((n,k)))
  H = abs(random.standard_normal((k,m)))
  for i in range(0,100000):
    H=H*(W.conj().transpose()*V)/(W.conj().transpose()*W*H+10^(-9))
    W=W*(V*H.conj().transpose())/(W*H*H.conj().transpose()+10^(-9))
  return W, H

def create_P_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                         home_points_col_ind, away_points_col_ind):
  n = len(teams_dict)
  P_matrix = np.zeros((n, n))
  for row in data_np:
    i, j, points_ij, points_ji = indices_and_points(row, teams_dict, home_col_ind, away_col_ind,
                                                    home_points_col_ind, away_points_col_ind)
    if i != j:
      if points_ij > 0 or points_ji >0:
          P_matrix[j][i] += 1
          P_matrix[i][j] += 1

  return P_matrix

def create_Tf_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                         home_points_col_ind, away_points_col_ind):
  n = len(teams_dict)
  t = np.zeros(n)
  f = np.zeros((n,1))
  T_matrix = np.zeros((n,n))
  for row in data_np:
    i, j, points_ij, points_ji = indices_and_points(row, teams_dict, home_col_ind, away_col_ind,
                                                    home_points_col_ind, away_points_col_ind)
    t[i] += 1
    t[j] += 1
    f[i] += points_ij
    f[j] += points_ji
  for x in range(0,len(t)):
    T_matrix[x][x] = t[x]
  return T_matrix, f

def massey_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                         home_points_col_ind, away_points_col_ind):
  n = len(teams_dict)
  p = np.zeros((n,1))
  p_temp = np.zeros((n,1))
  r = np.zeros((n,1))
  r_temp = np.zeros((n,1))
  M_matrix = np.zeros((n,n))
  M_temp = np.zeros((n,n))
  for row in data_np:
    i, j, points_ij, points_ji = indices_and_points(row, teams_dict, home_col_ind, away_col_ind,
                                                    home_points_col_ind, away_points_col_ind)
    M_matrix[i][j] -= 1
    M_matrix[j][i] -= 1
    M_matrix[i][i] += 1
    M_matrix[j][j] += 1
    p[i] += points_ij - points_ji
  #replace last row
  for i in range(0,n):
    M_temp = M_matrix.copy()
    p_temp = p.copy()
    M_temp[i,] = np.ones((n))
    p_temp[i] = 0
    r_temp = np.linalg.solve(M_temp,p_temp)
    r += r_temp
    r[i] -= r_temp[i]
  return r/(n-1)

def recency_weight(df, game_date):
  # Convert the "date" column to datetime type
  df['date'] = pd.to_datetime(df['GAME_DATE'])

  # Get today's date
  #today = datetime.datetime.now().date()

  # Calculate the difference in months between each date and today's date
  df['day_diff'] = 0
  for index, row in df.iterrows():
    df['day_diff'].loc[index] = (game_date.date() - row.date.date()).days


  # Calculate the weight for each date as a function of the month difference
  # full weight games within the last 28 days
  df['weight'] = 1 - (df['day_diff'] - no_days_in_block) / 90
  df.loc[df['weight'] > 1, 'weight'] = 1
  df.loc[df['weight'] < 0, 'weight'] = 0

  # Drop the month_diff column
  df = df.drop(columns=['day_diff', 'date'])
  df = df.drop_duplicates()
  return df


def determine_o_and_d(games_df, rating_df, teams, home_team_col_name, away_team_col_name, 
                        home_stat_col_name, away_stat_col_name):
  
  data_np = games_df.to_numpy()

  teams_dict = create_items_dict(teams)
  home_col_ind = games_df.columns.get_loc(home_team_col_name)
  away_col_ind = games_df.columns.get_loc(away_team_col_name)
  home_points_col_ind = games_df.columns.get_loc(home_stat_col_name)
  away_points_col_ind = games_df.columns.get_loc(away_stat_col_name)

  P_matrix = create_P_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                          home_points_col_ind, away_points_col_ind)

  T_matrix, f_matrix = create_Tf_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                          home_points_col_ind, away_points_col_ind)

  score_range = massey_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                          home_points_col_ind, away_points_col_ind)
  max_margin = score_range.max()
  min_margin = score_range.min()

  rating_arr = rating_df.Rating.to_numpy()

  scaler_rating = preprocessing.MinMaxScaler(feature_range =(min_margin, max_margin))
  rating_df['Rating_scaled']=scaler_rating.fit_transform(rating_arr.reshape(-1,1))

  r = np.zeros((len(rating_df),1))

  for index, row in rating_df.iterrows():
    r[teams_dict[row.team]] = row.Rating_scaled

  d = np.zeros((len(teams),1), dtype=float)
  o = np.zeros((len(teams),1), dtype=float)
    
  TplusP = np.add(T_matrix,P_matrix)
  Tr = np.dot(T_matrix,r)
  Trminusf = np.subtract(Tr,f_matrix)
  d = np.linalg.solve(TplusP,Trminusf)
  o = r - d

  rating_df['rat'] = 0
  rating_df['off'] = 0
  rating_df['def'] = 0

  for index, row in rating_df.iterrows():
    rating_df['rat'].loc[index] = r[teams_dict[row.team]]
    rating_df['off'].loc[index] = o[teams_dict[row.team]]
    rating_df['def'].loc[index] = d[teams_dict[row.team]]
  return rating_df

def lrmc_o_and_d(games_df, rating_df, teams, home_team_col_name, away_team_col_name, 
                        home_stat_col_name, away_stat_col_name):
  
  data_np = games_df.to_numpy()

  teams_dict = create_items_dict(teams)
  home_col_ind = games_df.columns.get_loc(home_team_col_name)
  away_col_ind = games_df.columns.get_loc(away_team_col_name)
  home_points_col_ind = games_df.columns.get_loc(home_stat_col_name)
  away_points_col_ind = games_df.columns.get_loc(away_stat_col_name)

  P_matrix = create_P_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                          home_points_col_ind, away_points_col_ind)

  T_matrix, f_matrix = create_Tf_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                          home_points_col_ind, away_points_col_ind)

  score_range = massey_matrix(data_np, teams_dict, home_col_ind, away_col_ind, 
                          games_df.columns.get_loc('home_score'), games_df.columns.get_loc('away_score'))
  max_margin = score_range.max()
  min_margin = score_range.min()

  rating_arr = rating_df.Rating.to_numpy()

  scaler_rating = preprocessing.MinMaxScaler(feature_range =(min_margin, max_margin))
  rating_df['Rating_scaled']=scaler_rating.fit_transform(rating_arr.reshape(-1,1))

  r = np.zeros((len(rating_df),1))

  for index, row in rating_df.iterrows():
    r[teams_dict[row.team]] = row.Rating_scaled

  d = np.zeros((len(teams),1), dtype=float)
  o = np.zeros((len(teams),1), dtype=float)
    
  TplusP = np.add(T_matrix,P_matrix)
  Tr = np.dot(T_matrix,r)
  Trminusf = np.subtract(Tr,f_matrix)
  d = np.linalg.solve(TplusP,Trminusf)
  o = r - d

  rating_df['rat'] = 0
  rating_df['off'] = 0
  rating_df['def'] = 0

  for index, row in rating_df.iterrows():
    rating_df['rat'].loc[index] = r[teams_dict[row.team]]
    rating_df['off'].loc[index] = o[teams_dict[row.team]]
    rating_df['def'].loc[index] = d[teams_dict[row.team]]
  return rating_df


def POSS_rank(games_df, rating_df, teams, home_team_col_name, away_team_col_name,
                 home_stat_col_name, away_stat_col_name):
    data_np = games_df.to_numpy()

    teams_dict = create_items_dict(teams)
    home_col_ind = games_df.columns.get_loc(home_team_col_name)
    away_col_ind = games_df.columns.get_loc(away_team_col_name)
    home_points_col_ind = games_df.columns.get_loc(home_stat_col_name)
    away_points_col_ind = games_df.columns.get_loc(away_stat_col_name)

    score_range = massey_matrix(data_np, teams_dict, home_col_ind, away_col_ind,
                                home_points_col_ind, away_points_col_ind)
    max_margin = score_range.max()
    min_margin = score_range.min()

    rating_arr = rating_df.Rating.to_numpy()

    scaler_rating = preprocessing.MinMaxScaler(feature_range=(min_margin, max_margin))
    rating_df['Rating_scaled'] = scaler_rating.fit_transform(rating_arr.reshape(-1, 1))

    return rating_df

def kde_from_sample(sample, qs):
    """Make a kernel density estimate from a sample."""
    kde = gaussian_kde(sample)
    ps = kde(qs)
    pmf = Pmf(ps, qs)
    pmf.normalize()
    return pmf

def make_uniform(qs, name=None, **options):
    """Make a Pmf that represents a uniform distribution."""
    pmf = Pmf(1.0, qs, **options)
    pmf.normalize()
    if name:
        pmf.index.name = name
    return pmf

def update_norm(prior, data):
    """Update the prior based on data."""
    mu_mesh, sigma_mesh, data_mesh = np.meshgrid(
        prior.columns, prior.index, data)
    
    densities = norm(mu_mesh, sigma_mesh).pdf(data_mesh)
    likelihood = densities.prod(axis=2)
    
    posterior = prior * likelihood
    normalize(posterior)

    return posterior

def update_norm_summary(prior, n, m, s):
    """Update a normal distribution using summary statistics."""
    mu_mesh, sigma_mesh = np.meshgrid(prior.columns, prior.index)
    
    like1 = norm(mu_mesh, sigma_mesh/np.sqrt(n)).pdf(m)
    like2 = chi2(n-1).pdf(n * s**2 / sigma_mesh**2)
    
    posterior = prior * like1 * like2
    normalize(posterior)
    
    return posterior

def prob_gt(pmf1, pmf2):
    """Compute the probability of superiority."""
    total = 0
    for q1, p1 in pmf1.items():
        for q2, p2 in pmf2.items():
            if q1 > q2:
                total += p1 * p2
    return total

def update_normal(pmf, data):
    """Update Pmf with a Norm likelihood."""
    likelihood = sts.norm.pdf(data, pmf)
    pmf *= likelihood
    pmf.normalize()

def common_opp(game_df, team_a, team_b):
    opponents_a = game_df['away_team'][game_df['home_team'] == team_a].tolist() + game_df['home_team'][game_df['away_team'] == team_a].tolist()
    opponents_b = game_df['away_team'][game_df['home_team'] == team_b].tolist() + game_df['home_team'][game_df['away_team'] == team_b].tolist()
    opponents_a = set(opponents_a)
    common_opponents = opponents_a.intersection(opponents_b)
    return common_opponents

def transition_prob(game_df, team_a, team_b, hfa):
    
    opponents_a = game_df['away_team'][game_df['home_team'] == team_a].tolist() + game_df['home_team'][game_df['away_team'] == team_a].tolist()
    opponents_b = game_df['away_team'][game_df['home_team'] == team_b].tolist() + game_df['home_team'][game_df['away_team'] == team_b].tolist()
    opponents_a = set(opponents_a)
    common_opponents = opponents_a.intersection(opponents_b)

    a_list = []
    b_list = []
    for index, game in game_df.loc[((game_df['home_team'].isin([team_a,team_b])) |
                                      game_df['away_team'].isin([team_a,team_b]))].iterrows():

        margin = game['home_score'] - game['away_score']
        
        if (game['home_team'] == team_a) & (game['away_team'] == team_b):
            a_list.append(margin + hfa)
            b_list.append(-margin - hfa)
        elif (game['home_team'] == team_b) & (game['away_team'] == team_a):
            a_list.append(-margin - hfa)
            b_list.append(margin + hfa)
        elif (game['home_team'] == team_a) & (game['away_team'] in common_opponents):
            a_list.append(margin + hfa)
        elif (game['home_team'] == team_b) & (game['away_team'] in common_opponents):
            b_list.append(margin + hfa)
        elif (game['away_team'] == team_a) & (game['home_team'] in common_opponents):
            a_list.append(-margin - hfa)
        elif (game['away_team'] == team_b) & (game['home_team'] in common_opponents):
            b_list.append(-margin - hfa)

    return a_list, b_list
