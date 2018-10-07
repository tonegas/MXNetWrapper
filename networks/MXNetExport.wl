(* ::Package:: *)

(*Loading the needed packages*)
<<MXNetLink`;
<<NeuralNetworks`;
<<GeneralUtilities`;
(*Support function*)
FixMXNnet[str_]:=StringJoin[{"arg:",str}];
(* Functions *)
MXNetNetworkExport[net_,networkStr_]:=Module[{numberOfEpochs=0,plan},
Export[StringJoin[{networkStr,"-symbol.json"}],ToMXJSON[net][[1]],"String"];
plan=ToMXPlan[net];
NDArrayExport[StringJoin[{networkStr,"-",StringPadLeft[ToString[numberOfEpochs],4,"0"],".params"}],NDArrayCreate/@KeyMap[FixMXNnet,plan["ArgumentArrays"]]]
];
