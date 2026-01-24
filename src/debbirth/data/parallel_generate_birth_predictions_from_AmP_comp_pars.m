%% Clear workspace
% Delete an existing progress bar in case it was not properly delete before

clear all
format long

%% Define paths to files

allSpeciesFolder = "C:\Users\diogo\OneDrive - Universidade de Lisboa\Terraprima\DEB Resources\DEBtool\AmPdata\species";
saveFolder = 'C:\Users\diogo\OneDrive - Universidade de Lisboa\Terraprima\Code\DEB Constraint Prediction\deb-birth-prediction\data\raw';

%% Get list of species

speciesList = getAllSpeciesNames(allSpeciesFolder);
numSpecies = length(speciesList);
numPointsPerSpecies = 20;
numPoints = numSpecies * numPointsPerSpecies;

%% Initialize table
parameterCols = {'g', 'k', 'v_Hb', 'f'};
columnNames = {
    'generator_species', 'point_id',  ...                          % info
     parameterCols{:}, ...                      % parameters
    'lb', 'tb', 'lb<f', 'k*vHb<c', 'reached_birth', ...        % outputs
    'success', 'execution_time', 'error_type', 'error_message', ... % logs
    };
numCols = length(columnNames);
varTypes = {
    'string', 'int32', ...
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
noiseLevel = 0.75; 
saveResultsTableEvery = 200;

% Max execution time per species
maxTime = 0.25; % in minutes
maxTime = maxTime * 60; % convert to seconds

printProgress = true;

% Output file
snl = strrep(sprintf('%.3f', noiseLevel), '.', 'p');  

outputFileName = [saveFolder '\' sprintf('AmP_comp_pars_noise_%s_seed_%d.csv', snl, seed)];


%% Set up parallel pool
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes');
end
numWorkers = pool.NumWorkers;
parfevalOnAll(@rng, 0, seed);

% Initialize variables
i = 1; % Index of species to submit
inProgressFutures = struct('future', {}, 'i', {}, 'speciesName', {}, 'startTime', {});

%% Start the processing loop
while i <= numPoints || ~isempty(inProgressFutures)
    % Submit new tasks if workers are available
    while length(inProgressFutures) < numWorkers && i <= numPoints
        speciesPoint = mod(i, numPointsPerSpecies) + 1;
        speciesIdx = floor(i / numPointsPerSpecies) + 1;
        speciesName = speciesList{speciesIdx};

        % Store info
        predictionsTable{i, 'generator_species'} = string(speciesName);
        predictionsTable{i, 'point_id'} = speciesPoint;
        % Generate random parameters
        try
            [genPar, success] = generateParameters(speciesName, allSpeciesFolder, noiseLevel);
            if ~success
                predictionsTable{i, 'success'} = false;
                predictionsTable{i, 'error_type'} = "par_gen_failed";
                predictionsTable{i, 'error_message'} = "Could not generate parameters to meet trivial birth constraint.";
                i = i + 1;
                continue
            end
            % Store parameters
            for p=1:length(parameterCols)
                parName = parameterCols{p};
                predictionsTable{i, parName} = genPar.(parName);
            end

            % Submit parfeval task
            fut = parfeval(pool, @runBirthSimulation, 2, genPar);
            % Record the future, species name, start time
            startTime = tic;
            nFutures = length(inProgressFutures);
            inProgressFutures(nFutures+1).future = fut;
            inProgressFutures(nFutures+1).i = i;
            inProgressFutures(nFutures+1).speciesName = speciesName;
            inProgressFutures(nFutures+1).speciesPoint = speciesPoint;
            inProgressFutures(nFutures+1).startTime = startTime;
        catch ME
            error_message = ME.message;
            predictionsTable{i, 'success'} = false;
            predictionsTable{i, 'error_type'} = "load_error";
            predictionsTable{i, 'error_message'} = string(error_message);
        end

        if printProgress
            fprintf('[%5d / %d | %50s] SUBMIT \n', i, numPoints, speciesName)
        end
        i = i + 1;
        % Write results to .csv file every once in a while
        if mod(i, saveResultsTableEvery) == 0
            writetable(predictionsTable, outputFileName,'WriteRowNames',false);
            fprintf('[%5d / %d | %50s] SAVE: Table saved at %s\n', i, numPoints, speciesName, outputFileName);
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
                    fprintf('[%5d / %d | %50s] RESULT: %d %.2f \n', futInfo.i, numPoints, futInfo.speciesName, success, executionTime)
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
                    fprintf('[%5d / %d | %50s] ERROR: %s %.2f \n', futInfo.i, numPoints, futInfo.speciesName, error_message, executionTime)
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
                    fprintf('[%5d / %d | %50s] TIMEOUT: predict function took longer than %d seconds to execute. \n', futInfo.i, numPoints, futInfo.speciesName, maxTime)
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

%% Function to process each species
function [genPar, success] = generateParameters(speciesName, allSpeciesFolder, noiseLevel)
speciesFolder = fullfile(allSpeciesFolder, speciesName);
genPar = struct();
% Check if the species folder exists
if isfolder(speciesFolder)
    % Change directory to the species folder
    cd(speciesFolder);

    % Run mydata.m
    [~, ~, metaData, ~, ~] = feval(['mydata_' speciesName]);
    % Get parameters from .mat file
    resultsMatFilename = ['results_' speciesName '.mat'];
    if exist(resultsMatFilename, 'file')
        load(resultsMatFilename, "par", "metaPar")
    else
        [par, metaPar, ~] = feval(['pars_init_' speciesName], metaData);
    end
    cPar = parscomp_st(par);

    % Check that DEB model has an egg stage
    if strcmp(metaPar.model, {'stx', 'stf'})
        return
    end
    kvHb = 2;
    nTries = 0; 
    while kvHb > 1 && nTries < 100
        % Add variation to parameters
        genPar.k = addMultiplicativeNoise(cPar.k, noiseLevel);
        genPar.v_Hb = addMultiplicativeNoise(cPar.v_Hb, noiseLevel);
        genPar.g = addMultiplicativeNoise(cPar.g, noiseLevel);
        % Compute compound parameters
        kvHb = genPar.k * genPar.v_Hb;
        nTries = nTries + 1;
    end
    if kvHb > 1
        genPar = struct(); success = false;
        return
    end
    % Generate random f such that it meets the trivial condition k*v_Hb<f^3
    % if rand() > 0.5
    %     par.f = 1;
    % else
        f3 = kvHb + (1 - kvHb) * rand();
        genPar.f = f3^(1/3);
    % end
    success = true;
else
    error('Folder for species "%s" does not exist.', speciesName);
end
end


function [outputs, success] = runBirthSimulation(genPar)
outputs = struct( ...
    'lb', NaN, ...
    'tb', NaN, ...
    'lb_f', false, ...
    'k_vHb_c', false, ...
    'reached_birth', false ...
    );
success = false;

% Get compound parameters for computing length at birth
g = genPar.g; k = genPar.k; v_Hb = genPar.v_Hb; f = genPar.f;
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

function randomValue = addMultiplicativeNoise(value, noiseLevel)
noise = (1 - noiseLevel) + (2 * noiseLevel) * rand();
randomValue = value .* noise;
end

function speciesList = getAllSpeciesNames(allSpeciesFolder)
% Get a list of all files and folders in the specified directory
allFiles = dir(allSpeciesFolder);

% Filter the list to include only directories
isDir = [allFiles.isdir]; % Logical index for directories
speciesList = {allFiles(isDir).name}; % Extract names of directories

% Remove '.' and '..' from the list (current and parent directory)
speciesList = speciesList(~ismember(speciesList, {'.', '..'}));
end
