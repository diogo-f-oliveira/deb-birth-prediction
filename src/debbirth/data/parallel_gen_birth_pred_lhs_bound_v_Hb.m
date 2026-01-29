%% DEB birth dataset generation with regime-aware v_Hb sampling
% - Grids for g, k, f
% - For each (g,k,f) we sample v_Hb log-uniformly within a regime-aware bracket
% - Runs simulations in parallel via parfeval

%% Clear workspace
clear all
format long

%% Define paths to files
saveFolder = '..\..\..\data\raw';

%%  Parameter ranges 
g_pow_min = -3;
g_pow_max =  2;

k_pow_min = -4;
k_pow_max = 1;

f_min = 0.1;
f_max = 1.0;

% Optional symmetric expansion (in decades) beyond the bracket
%   0.0 disables expansion
%   0.5 expands by ~x3.16 on each side
%   1.0 expands by x10 on each side
delta_decades = 1;

% Total points
numPoints = 200000;

% Reproducibility: sampling happens in the main thread (before parfeval)
rng(0, 'twister');


%%  Space-filling sampling (LHS)

% Sample g,k,f in a space-filling way with Latin Hypercube in 3 dims.
% u(:,1)=g, u(:,2)=k, u(:,3)=f
u = lhsdesign(numPoints, 3, 'criterion','maximin','iterations',50);

% log-uniform g
ug = g_pow_min + u(:,1) * (g_pow_max - g_pow_min);
g_list = 10.^ug;

% log-uniform k
uk = k_pow_min + u(:,2) * (k_pow_max - k_pow_min);
k_list = 10.^uk;

% uniform f
f_list = f_min + u(:,3) * (f_max - f_min);

% v_Hb sampled conditional on (k,f)
vHb_list = zeros(numPoints,1);
for n = 1:numPoints
    vHb_list(n) = samplevHb(k_list(n), f_list(n), delta_decades);
end

%% Initialize table
parameterCols = {'g', 'k', 'v_Hb', 'f'};
columnNames = {
     parameterCols{:}, ...   % parameters
    'lb', 'tb', 'lb<f', 'k*vHb<c', 'reached_birth', ...        % outputs
    'success', 'execution_time', 'error_type', 'error_message', ... % logs
    };
numCols = length(columnNames);

varTypes = {
    'double', 'double', 'double', 'double', ...
    'double', 'double', 'logical', 'logical', 'logical', ...
    'logical', 'double', 'string', 'string', ...
    };

predictionsTable = table( ...
    'Size', [numPoints, numCols], ...
    'VariableTypes', varTypes, ...
    'VariableNames', columnNames, ...
    'RowNames', cellstr(string(1:numPoints)) ...
    );

%% Settings
seed = 42;
saveResultsTableEvery = 1000;

% Max execution time per simulation
maxTime = 0.1; % in minutes
maxTime = maxTime * 60; % convert to seconds

printProgress = false;
fmtNum = @(x) regexprep(sprintf('%.6g', x), {'\.', '-', '\+'}, {'p','m',''});

% Output file
fname = sprintf([ ...
    'sample4D_lhs_' ...
    'N_%d_' ...
    'g_1e%s_1e%s_' ...
    'k_%s_%s_' ...
    'f_%s_%s_' ...
    'ddec_%s' ...
    '.csv'], ...
    numPoints, ...
    fmtNum(g_pow_min), fmtNum(g_pow_max), ...
    fmtNum(k_pow_min), fmtNum(k_pow_max), ...
    fmtNum(f_min), fmtNum(f_max), ...
    fmtNum(delta_decades));

outputFileName = [saveFolder '\' fname];

%% Set up parallel pool
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes');
end
numWorkers = pool.NumWorkers;
parfevalOnAll(@rng, 0, seed);

%% Processing loop with parfeval
i = 1; % linear index over all points
inProgressFutures = struct('future', {}, 'i', {}, 'startTime', {});

while i <= numPoints || ~isempty(inProgressFutures)

    % Submit new tasks if workers are available
    while length(inProgressFutures) < numWorkers && i <= numPoints

        % Map linear index -> (iv, ig, ik, jf)
        % Dimensions order: [NvHb, Ng, Nk, Nf]
        g    = g_list(i);
        k    = k_list(i);
        f    = f_list(i);
        v_Hb = vHb_list(i);

        % Store parameters
        predictionsTable{i, 'g'}    = g;
        predictionsTable{i, 'k'}    = k;
        predictionsTable{i, 'f'}    = f;
        predictionsTable{i, 'v_Hb'} = v_Hb;

        % Submit task
        fut = parfeval(pool, @runBirthSimulation, 2, g, k, v_Hb, f);

        startTime = tic;
        nFutures = length(inProgressFutures);
        inProgressFutures(nFutures+1).future = fut;
        inProgressFutures(nFutures+1).i = i;
        inProgressFutures(nFutures+1).startTime = startTime;

        if printProgress
            fprintf('[%8d / %8d] SUBMIT\n', i, numPoints);
        end

        i = i + 1;

        % Periodic save
        if mod(i, saveResultsTableEvery) == 0
            writetable(predictionsTable, outputFileName, 'WriteRowNames', true);
            fprintf('[%8d / %8d] SAVE: %s\n', i, numPoints, outputFileName);
        end
    end

    % Check futures for completion or timeout
    idx = 1;
    while idx <= length(inProgressFutures)
        futInfo = inProgressFutures(idx);

        if strcmp(futInfo.future.State, 'finished')
            try
                [outputs, success] = fetchOutputs(futInfo.future);
                executionTime = toc(futInfo.startTime);

                if printProgress
                    fprintf('[%8d / %8d] RESULT: %d  %.2f s\n', futInfo.i, numPoints, success, executionTime);
                end

                % Store outputs
                predictionsTable{futInfo.i, 'lb'} = outputs.lb;
                predictionsTable{futInfo.i, 'tb'} = outputs.tb;
                predictionsTable{futInfo.i, 'lb<f'} = outputs.lb_f;
                predictionsTable{futInfo.i, 'k*vHb<c'} = outputs.k_vHb_c;
                predictionsTable{futInfo.i, 'reached_birth'} = outputs.reached_birth;

                % Logs
                predictionsTable{futInfo.i, 'execution_time'} = executionTime;
                predictionsTable{futInfo.i, 'success'} = success;
                predictionsTable{futInfo.i, 'error_type'} = "none";

            catch ME
                if isempty(futInfo.future.Error)
                    error_message = ME.message;
                else
                    error_message = futInfo.future.Error.message;
                end
                executionTime = toc(futInfo.startTime);

                if printProgress
                    fprintf('[%8d / %8d] ERROR: %s  %.2f s\n', futInfo.i, numPoints, error_message, executionTime);
                end

                predictionsTable{futInfo.i, 'execution_time'} = executionTime;
                predictionsTable{futInfo.i, 'success'} = false;
                predictionsTable{futInfo.i, 'error_type'} = "execution_error";
                predictionsTable{futInfo.i, 'error_message'} = string(error_message);
            end

            % Remove future
            inProgressFutures(idx) = [];

        else
            % Timeout check
            elapsedTime = toc(futInfo.startTime);
            if elapsedTime > maxTime
                cancel(futInfo.future);

                if printProgress
                    fprintf('[%8d / %8d] TIMEOUT: > %d s\n', futInfo.i, numPoints, maxTime);
                end

                predictionsTable{futInfo.i, 'execution_time'} = maxTime;
                predictionsTable{futInfo.i, 'success'} = false;
                predictionsTable{futInfo.i, 'error_type'} = "time_limit_reached";
                predictionsTable{futInfo.i, 'error_message'} = "Maximum execution time exceeded";

                inProgressFutures(idx) = [];
            else
                idx = idx + 1;
            end
        end
    end

    pause(0.05);
end

%% Final save
writetable(predictionsTable, outputFileName, 'WriteRowNames', true);
fprintf('Table saved in %s\n', outputFileName);

%% ============================================================
%  Local functions
%  ============================================================

function v_Hb = samplevHb(k, f, delta_decades)
%SAMPLEVHB Regime-aware log-uniform sampling of v_Hb.
%
% Boundary bracket:
%   - k < 1:  v_Hb in [f^3,           f^3/k]
%   - k > 1:  v_Hb in [f^3/k^3,       f^3/k]
%
% Optionally expands bracket by +/- delta_decades in log10 space.
%
% Inputs:
%   k, f           : DEB parameters (assumed positive)
%   delta_decades  : expansion (>=0). 0 disables expansion.
%
% Output:
%   v_Hb           : sampled value

    if ~(k > 0) || ~(f > 0)
        error('samplevHb:InvalidInputs', 'k and f must be > 0');
    end

    % Regime-aware bracket
    if k < 1
        v_lo = f^3;
        v_hi = f^3 / k;
    else
        v_lo = f^3 / (k^3);
        v_hi = f^3 / k;
    end

    % Ensure ordering (just in case)
    v_lo0 = min(v_lo, v_hi);
    v_hi0 = max(v_lo, v_hi);
    v_lo = v_lo0;
    v_hi = v_hi0;

    % Optional symmetric expansion in decades
    if nargin < 3
        delta_decades = 0.0;
    end
    if delta_decades > 0
        v_lo = v_lo / (10^delta_decades);
        v_hi = v_hi * (10^delta_decades);
    end

    % Safety: strictly positive, and lo < hi
    v_lo = max(v_lo, realmin('double'));
    if v_hi <= v_lo
        v_hi = v_lo * 10;
    end

    % Log-uniform sample
    t = rand();
    v_Hb = v_lo * (v_hi / v_lo) ^ t;
end

function [outputs, success] = runBirthSimulation(g, k, v_Hb, f)
outputs = struct( ...
    'lb', NaN, ...
    'tb', NaN, ...
    'lb_f', false, ...
    'k_vHb_c', false, ...
    'reached_birth', false ...
    );
success = false;

% Get compound parameters for computing length at birth
pars_lb = [g, k, v_Hb];

% Compute length at birth
[lb, info] = get_lb(pars_lb, f);
if ~info; return; end
outputs.lb = lb;
outputs.lb_f = lb < f;
outputs.k_vHb_c = k * v_Hb < f/ (g + f) * lb^2 * (g + lb);
outputs.reached_birth = outputs.lb_f & outputs.k_vHb_c;

% Compute scaled age at birth
[tb, ~, info] = get_tb(pars_lb, f, lb);
if ~info; return; end
outputs.tb = tb;

success = true;
end
