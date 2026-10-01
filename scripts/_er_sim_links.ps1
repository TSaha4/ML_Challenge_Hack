# Hard-link unchanged inputs into the hard-mode (hidden-S1) run so no extra disk is used.
$r = 'C:\Users\Admin\Downloads\ML_Challenge_Hack'
function Link($dst, $src) {
    if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Force }
    New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null
}
foreach ($f in 'train_source2.tsv', 'train_source3.tsv') {
    Link "$r\student_resource_sim\dataset\train\$f" "$r\student_resource\dataset\train\$f"
}
foreach ($f in 'test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv') {
    Link "$r\student_resource_sim\dataset\test\$f" "$r\student_resource\dataset\test\$f"
}
foreach ($f in 'test_s1.parquet', 'test_tg.parquet', 'test_s1stats.parquet', 'translit_all.json') {
    Link "$r\artifacts\er_sim\$f" "$r\artifacts\er_fixed\$f"
}
foreach ($d in 'test_cands', 'test_sibs') {
    New-Item -ItemType Directory -Force "$r\artifacts\er_sim\$d" | Out-Null
    Get-ChildItem "$r\artifacts\er_fixed\$d" -File | ForEach-Object { Link "$r\artifacts\er_sim\$d\$($_.Name)" $_.FullName }
}
New-Item -ItemType Directory -Force "$r\artifacts\er_sim\nn" | Out-Null
Link "$r\artifacts\er_sim\nn\matcher.pt" "$r\artifacts\er_fixed\nn\matcher.pt"
'links done'
