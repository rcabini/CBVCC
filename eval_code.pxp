<?php
function parse_csv($file) {
    $data = [];
    if (($handle = fopen($file, 'r')) !== FALSE) {
        while (($row = fgetcsv($handle, 1000, ",")) !== FALSE) {
            $data[$row[0]] = $row[1];
        }
        fclose($handle);
    }
    return $data;
}

function calculate_metrics($groundTruth, $predictions) {
    $TP = $FP = $FN = $TN = 0;
    $thresholds = array_unique(array_values($predictions));
    sort($thresholds);
    $rocPoints = [];

    foreach ($thresholds as $th) {
        $TP = $FP = $FN = $TN = 0;

        foreach ($groundTruth as $id => $trueClass) {
            if (!isset($predictions[$id])){
				$missing_pred = 1;
				continue;
			};

            $predictedClass = floatval($predictions[$id]);
            $trueClass = floatval($trueClass);

            // Calculate TP, FP, FN, TN
            if ($trueClass >= 0.5 && $predictedClass >= $th) $TP++;
            elseif ($trueClass >= 0.5 && $predictedClass < $th) $FN++;
            elseif ($trueClass < 0.5 && $predictedClass >= $th) $FP++;
            elseif ($trueClass < 0.5 && $predictedClass < $th) $TN++;
        }

        // Calculate TPR and FPR for this threshold
        $tpr = ($TP + $FN > 0) ? $TP / ($TP + $FN) : 0;
        $fpr = ($FP + $TN > 0) ? $FP / ($FP + $TN) : 0;

        $rocPoints[] = ['FPR' => $fpr, 'TPR' => $tpr];
    }

    // Calculate AUC using the trapezoidal rule
    $auc = 0;
    for ($i = 1; $i < count($rocPoints); $i++) {
        $x1 = $rocPoints[$i - 1]['FPR'];
        $y1 = $rocPoints[$i - 1]['TPR'];
        $x2 = $rocPoints[$i]['FPR'];
        $y2 = $rocPoints[$i]['TPR'];
        $auc += abs($x2 - $x1) * ($y1 + $y2) / 2;
    }

    // Calculate metrics for the default threshold (0.5)
    $TP = $FP = $FN = $TN = 0;
    $defaultThreshold = 0.5;
    foreach ($groundTruth as $id => $trueClass) {
        if (!isset($predictions[$id])) continue;

        $predictedClass = floatval($predictions[$id]);
        $trueClass = floatval($trueClass);

        if ($trueClass >= 0.5 && $predictedClass >= $defaultThreshold) $TP++;
        elseif ($trueClass >= 0.5 && $predictedClass < $defaultThreshold) $FN++;
        elseif ($trueClass < 0.5 && $predictedClass >= $defaultThreshold) $FP++;
        elseif ($trueClass < 0.5 && $predictedClass < $defaultThreshold) $TN++;
    }

    $sensitivity = ($TP + $FN > 0) ? $TP / ($TP + $FN) : 0;
    $specificity = ($TN + $FP > 0) ? $TN / ($TN + $FP) : 0;
    $balancedAccuracy = ($sensitivity + $specificity) / 2;
	$accuracy = ($TP+$TN) / ($TP+$TN+$FP+$FN);
	if(($TP+$FP) > 0){
	$precision = ($TP) / ($TP+$FP);
	}
	else{
		$precision = 0;
	}
	if(($TP+$FN) > 0){
		$recall = ($TP) / ($TP+$FN);
	}
	else{
		$recall = 0;
	}	

    return [
        'TP' => $TP,
        'FP' => $FP,
        'FN' => $FN,
        'TN' => $TN,
        'Sensitivity' => round($sensitivity, 4),
        'Specificity' => round($specificity, 4),
		'Accuracy' => round($accuracy, 4),
		'Precision' => round($precision, 4),
		'Recall' => round($recall, 4),
        'Balanced Accuracy' => round($balancedAccuracy, 4),
        'AUC' => round($auc, 4)
    ];
}

//---- main ----//
$missing_pred = 0;
$groundTruthFile = 'phase_1_GT.csv';
$predictionsFile = 'your_file.csv';
$groundTruth = parse_csv($groundTruthFile);
$predictions = parse_csv($predictionsFile);
