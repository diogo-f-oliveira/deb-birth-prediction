%% Clear workspace
% Delete an existing progress bar in case it was not properly delete before

clear all
format long

%% Define paths to files
saveFolder = '..\..\..\data\raw';

%% Parameter specifications (each can be fixed or ranged)
% mode: 'fixed' or 'range'
% scale (only for 'range'): 'log' or 'lin'

spec.g    = struct('mode','range','min',1e-3,'max',1e2,'n',50,'scale','log');
spec.v_Hb = struct('mode','range','min',1e-10,'max',1e1,'n',100,'scale','log');
spec.k    = struct('mode','range','min',1e-4,'max',1e1,'n',50,'scale','log');
spec.f    = struct('mode','range','min',0.1,'max',1,'n',10,'scale','lin');

% Build vectors
g_vec    = makeVec(spec.g);
vHb_vec  = makeVec(spec.v_Hb);
k_vec    = makeVec(spec.k);
f_vec    = makeVec(spec.f);

% N-D grid (works for any combination of fixed/range)
% NOTE: ndgrid generalizes meshgrid to N dimensions.
[G, K, VHB, F] = ndgrid(g_vec, k_vec, vHb_vec, f_vec);

numPoints = numel(G);


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
saveResultsTableEvery = 1000;

% Max execution time per species
maxTime = 0.25; % in minutes
maxTime = maxTime * 60; % convert to seconds

printProgress = true;

% Output file
fmtNum = @(x) regexprep(sprintf('%.6g', x), {'\.', '-', '\+'}, {'p','m',''});
tag = @(s) makeTag(s, fmtNum);

fname = sprintf('grid_g_%s_k_%s_vHb_%s_f_%s.csv', ...
    tag(spec.g), tag(spec.k), tag(spec.v_Hb), tag(spec.f));

outputFileName = [saveFolder '\' fname];



%% Set up parallel pool
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes');
end
numWorkers = pool.NumWorkers;

% Initialize variables
i = 1; % Index of species to submit
inProgressFutures = struct('future', {}, 'i', {}, 'speciesName', {}, 'startTime', {});

%% Start the processing loop
while i <= numPoints || ~isempty(inProgressFutures)
    % Submit new tasks if workers are available
    while length(inProgressFutures) < numWorkers && i <= numPoints
        
        % Get parameters
        g    = G(i);
        k    = K(i);
        v_Hb = VHB(i);
        f    = F(i);

        % Store parameters
        predictionsTable{i, 'g'}    = g;
        predictionsTable{i, 'k'}    = k;
        predictionsTable{i, 'v_Hb'} = v_Hb;
        predictionsTable{i, 'f'}    = f;
        
        % Submit parfeval task
        fut = parfeval(pool, @runBirthSimulation, 2, g, k, v_Hb, f);

        % Record the future start time
        startTime = tic;
        nFutures = length(inProgressFutures);
        inProgressFutures(nFutures+1).future = fut;
        inProgressFutures(nFutures+1).i = i;
        inProgressFutures(nFutures+1).startTime = startTime;

        if printProgress
            fprintf('[%6d / %6d] SUBMIT \n', i, numPoints)
        end
        i = i + 1;
        % Write results to .csv file every once in a while
        if mod(i, saveResultsTableEvery) == 0
            writetable(predictionsTable, outputFileName,'WriteRowNames',true);
            fprintf('[%6d / %6d] SAVE: Table saved at %s\n', i, numPoints, outputFileName);
        end
    end

    % Check futures for completion or timeout
    idx = 1;
    while idx <= length(inProgressFutures)
        futInfo = inProgressFutures(idx);
        if strcmp(futInfo.future.State, 'finished')
            % Fetch outputs
            try
                [outputs, success] = fetchOutputs(futInfo.future);
                executionTime = toc(futInfo.startTime);
                if printProgress
                    fprintf('[%6d / %6d] RESULT: %d %.2f \n', futInfo.i, numPoints, success, executionTime)
                end
                % Store outputs
                predictionsTable{futInfo.i, 'lb'} = outputs.lb;
                predictionsTable{futInfo.i, 'tb'} = outputs.tb;
                predictionsTable{futInfo.i, 'lb<f'} = outputs.lb_f;
                predictionsTable{futInfo.i, 'k*vHb<c'} = outputs.k_vHb_c;
                predictionsTable{futInfo.i, 'reached_birth'} = outputs.reached_birth;
                % Store logs
                predictionsTable{futInfo.i, 'execution_time'} = executionTime;
                predictionsTable{futInfo.i, 'success'} = success;
                predictionsTable{futInfo.i, 'error_type'} = "none";

            catch ME
                % Handle error
                if isempty(futInfo.future.Error)
                    error_message = ME.message;
                else
                    error_message = futInfo.future.Error.message;
                end
                executionTime = toc(futInfo.startTime);
                if printProgress
                    fprintf('[6d / %6d] ERROR: %s %.2f \n', futInfo.i, numPoints, error_message, executionTime)
                end
                predictionsTable{futInfo.i, 'execution_time'} = executionTime;
                predictionsTable{futInfo.i, 'success'} = false;
                predictionsTable{futInfo.i, 'error_type'} = "execution_error";
                predictionsTable{futInfo.i, 'error_message'} = string(error_message);
            end
            % Remove future from in-progress list
            inProgressFutures(idx) = [];
        else
            % Check for timeout
            elapsedTime = toc(futInfo.startTime);
            if elapsedTime > maxTime
                cancel(futInfo.future);
                if printProgress
                    fprintf('[%6d / %6d] TIMEOUT: predict function took longer than %d seconds to execute. \n', futInfo.i, numPoints, maxTime)
                end
                predictionsTable{futInfo.i, 'execution_time'} = maxTime;
                predictionsTable{futInfo.i, 'success'} = false;
                predictionsTable{futInfo.i, 'error_type'} = "time_limit_reached";
                predictionsTable{futInfo.i, 'error_message'} = "Maximum execution time exceeded";
                % Remove future from in-progress list
                inProgressFutures(idx) = [];
            else
                idx = idx + 1;
            end
        end
    end

    % Pause for a short time to avoid busy waiting
    pause(0.1);
end

%% Write results to a .csv file
writetable(predictionsTable, outputFileName,'WriteRowNames',true);
fprintf('Table saved in %s\n', outputFileName);

%% Helper functions
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
[lb, info] = get_lb2(pars_lb, f);
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

function vec = makeVec(s)
    switch lower(s.mode)
        case 'fixed'
            vec = s.value;

        case 'range'
            if ~isfield(s,'scale'), s.scale = 'lin'; end
            if s.n <= 1
                vec = s.min;  % degenerate range -> single point
                return
            end

            switch lower(s.scale)
                case 'log'
                    vec = logspace(log10(s.min), log10(s.max), s.n);
                case 'lin'
                    vec = linspace(s.min, s.max, s.n);
                otherwise
                    error("Unknown scale '%s'. Use 'log' or 'lin'.", s.scale);
            end

        otherwise
            error("Unknown mode '%s'. Use 'fixed' or 'range'.", s.mode);
    end
end

function t = makeTag(s, fmtNum)
    switch lower(s.mode)
        case 'fixed'
            t = ['fixed_' fmtNum(s.value)];
        case 'range'
            if ~isfield(s,'scale'), s.scale = 'lin'; end
            t = sprintf('%s_%s_%s_N%d', lower(s.scale), fmtNum(s.min), fmtNum(s.max), s.n);
        otherwise
            error("Unknown mode '%s'.", s.mode);
    end
end


