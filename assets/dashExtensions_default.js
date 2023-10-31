window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
                const {
                    color_key,
                    current_cell,
                    fillOpacity,
                    ftu_color,
                    filter_vals
                } = context.hideout;
                if (current_cell) {
                    if (current_cell === 'cluster') {
                        if (current_cell in feature.properties) {
                            var cell_value = feature.properties.Cluster;
                            cell_value = (Number(cell_value)).toFixed(1);
                        } else {
                            cell_value = Number.Nan;
                        }
                    } else if (current_cell === 'max') {
                        // Extracting all the cell values for a given FTU/Spot
                        if ("Main_Cell_Types" in feature.properties) {
                            var cell_values = feature.properties.Main_Cell_Types;
                            // Initializing some comparison values
                            var cell_value = 0.0;
                            var use_cell_value = 0.0;
                            var cell_idx = -1.0;
                            // Iterating through each cell type in cell values
                            for (var key in cell_values) {
                                cell_idx += 1.0;
                                var test_val = cell_values[key];
                                // If the test value is greater than the cell_value, replace cell value with that test value
                                if (test_val > cell_value) {
                                    cell_value = test_val;
                                    use_cell_value = cell_idx;
                                }
                            }
                            cell_value = (use_cell_value).toFixed(1);
                        } else {
                            cell_value = Number.Nan;
                        }
                    } else if ('Main_Cell_Types' in feature.properties) {
                        if (current_cell in feature.properties.Main_Cell_Types) {

                            var cell_value = feature.properties.Main_Cell_Types[current_cell];
                            if (cell_value == 1) {
                                cell_value = (cell_value).toFixed(1);
                            } else if (cell_value == 0) {
                                cell_value = (cell_value).toFixed(1);
                            }
                        } else if (current_cell in feature.properties) {
                            var cell_value = feature.properties[current_cell];
                        } else if (current_cell.includes('_')) {

                            var split_cell_value = current_cell.split("_");
                            var main_cell_value = split_cell_value[0];
                            var sub_cell_value = split_cell_value[1];

                            var cell_value = feature.properties.Main_Cell_Types[main_cell_value];
                            cell_value *= feature.properties.Cell_States[main_cell_value][sub_cell_value];

                            if (cell_value == 1) {
                                cell_value = (cell_value).toFixed(1);
                            } else if (cell_value == 0) {
                                cell_value = (cell_value).toFixed(1);
                            }
                        } else {
                            var cell_value = Number.Nan;
                        }
                    } else {
                        var cell_value = Number.Nan;
                    }
                } else {
                    var cell_value = Number.Nan;
                }

                var style = {};
                if (cell_value == cell_value) {
                    const fillColor = color_key[cell_value];

                    style.fillColor = fillColor;
                    style.fillOpacity = fillOpacity;
                    style.color = ftu_color;

                } else {
                    style.color = ftu_color;
                }

                return style;
            }

            ,
        function1: function(feature, context) {
            const {
                color_key,
                current_cell,
                fillOpacity,
                ftu_color,
                filter_vals
            } = context.hideout;

            if (current_cell) {
                if ("Main_Cell_Types" in feature.properties) {
                    if (current_cell in feature.properties.Main_Cell_Types) {
                        var cell_value = feature.properties.Main_Cell_Types[current_cell];
                        if (cell_value >= filter_vals[0]) {
                            if (cell_value <= filter_vals[1]) {
                                return true;
                            } else {
                                return false;
                            }
                        } else {
                            return false;
                        }
                    } else if (current_cell in feature.properties) {
                        var cell_value = feature.properties[current_cell];
                        if (cell_value >= filter_vals[0]) {
                            if (cell_value <= filter_vals[1]) {
                                return true;
                            } else {
                                return false;
                            }
                        } else {
                            return false;
                        }
                    } else if (current_cell.includes('_')) {
                        var split_cell_value = current_cell.split("_");
                        var main_cell_value = split_cell_value[0];
                        var sub_cell_value = split_cell_value[1];

                        var cell_value = feature.properties.Main_Cell_Types[main_cell_value];
                        cell_value *= feature.properties.Cell_States[main_cell_value][sub_cell_value];
                        if (cell_value >= filter_vals[0]) {
                            if (cell_value <= filter_vals[1]) {
                                return true;
                            } else {
                                return false;
                            }
                        } else {
                            return false;
                        }

                    } else {
                        return true;
                    }
                } else if (current_cell in feature.properties) {
                    var cell_value = feature.properties[current_cell];
                    if (cell_value >= filter_vals[0]) {
                        if (cell_value <= filter_vals[1]) {
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return true;
                }
            } else {
                return true;
            }
        }

    }
});