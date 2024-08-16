window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(e, ctx) {
            ctx.map.flyTo([-120, 120], 1);
        },
        function1: function(e, ctx) {
            console.log(`Clicked coordinates: ${e.latlng}, map center: ${ctx.map.getCenter()}`);
        },
        function2: function(feature, context) {
                const {
                    color_key,
                    overlay_prop,
                    fillOpacity,
                    ftu_colors,
                    filter_vals
                } = context.hideout;

                var overlay_value = Number.Nan;
                if (overlay_prop && "user" in feature.properties) {
                    if (overlay_prop.name) {
                        if (overlay_prop.name in feature.properties.user) {
                            if (overlay_prop.value) {
                                if (overlay_prop.value in feature.properties.user[overlay_prop.name]) {
                                    if (overlay_prop.sub_value) {
                                        if (overlay_prop.sub_value in feature.properties.user[overlay_prop.name][overlay_prop.value]) {
                                            var overlay_value = feature.properties.user[overlay_prop.name][overlay_prop.value][overlay_prop.sub_value];
                                        } else {
                                            var overlay_value = Number.Nan;
                                        }
                                    } else {
                                        // TODO: Might have to do something here for non-aggregated String props
                                        var overlay_value = feature.properties.user[overlay_prop.name][overlay_prop.value];
                                    }
                                } else if (overlay_prop.value === "max") {
                                    // Finding max represented sub-value
                                    var overlay_value = Number.Nan;
                                    var test_value = 0.0;
                                    for (var key in feature.properties.user[overlay_prop.name]) {
                                        var tester = feature.properties.user[overlay_prop.name][key];
                                        if (tester > test_value) {
                                            test_value = tester;
                                            overlay_value = key;
                                        }
                                    }
                                } else {
                                    var overlay_value = Number.Nan;
                                }
                            } else {
                                var overlay_value = feature.properties.user[overlay_prop.name];
                            }
                        } else {
                            var overlay_value = Number.Nan;
                        }
                    } else {
                        var overlay_value = Number.Nan;
                    }
                } else {
                    var overlay_value = Number.Nan;
                }

                var style = {};
                if (overlay_value == overlay_value) {
                    if (overlay_value in color_key) {
                        const fillColor = color_key[overlay_value];
                        style.fillColor = fillColor;
                        style.fillOpacity = fillOpacity;
                    } else if (Number(overlay_value).toFixed(1) in color_key) {
                        const fillColor = color_key[Number(overlay_value).toFixed(1)];
                        style.fillColor = fillColor;
                        style.fillOpacity = fillOpacity;
                    }

                    if (feature.properties.name in ftu_colors) {
                        style.color = ftu_colors[feature.properties.name];
                    } else {
                        style.color = 'white';
                    }

                } else {
                    if (feature.properties.name in ftu_colors) {
                        style.color = ftu_colors[feature.properties.name];
                    } else {
                        style.color = 'white';
                    }
                    style.fillColor = "f00";
                }

                return style;
            }

            ,
        function3: function(feature, context) {
                const {
                    color_key,
                    overlay_prop,
                    fillOpacity,
                    ftu_colors,
                    filter_vals
                } = context.hideout;
                var return_feature = true;
                if (filter_vals) {
                    // If there are filters, use them
                    for (let i = 0; i < filter_vals.length; i++) {
                        // Iterating through filter_vals dict
                        var filter = filter_vals[i];

                        if (filter.name) {
                            // Checking if the filter name is in the feature
                            if (filter.name in feature.properties.user) {

                                if (filter.value) {
                                    if (filter.value in feature.properties.user[filter.name]) {
                                        if (filter.sub_value) {
                                            if (filter.sub_value in feature.properties.user[filter.name][filter.value]) {
                                                var test_val = feature.properties.user[filter.name][filter.value][filter.sub_value];
                                            } else {
                                                return_feature = return_feature & false;
                                            }
                                        } else {
                                            // TODO: Might have to do something here for non-aggregated String props
                                            var test_val = feature.properties.user[filter.name][filter.value];
                                        }
                                    } else if (filter.value === "max") {
                                        return_feature = return_feature & true;
                                    } else {
                                        return_feature = return_feature & false;
                                    }
                                } else {
                                    var test_val = feature.properties.user[filter.name];
                                }
                            } else {
                                return_feature = return_feature & false;
                            }
                        }

                        if (filter.range) {
                            if (typeof filter.range[0] === 'number') {
                                if (test_val < filter.range[0]) {
                                    return_feature = return_feature & false;
                                }
                                if (test_val > filter.range[1]) {
                                    return_feature = return_feature & false;
                                }
                            } else {
                                if (filter.range.includes(return_feature)) {
                                    return_feature = return_feature & true;
                                } else {
                                    return_feature = return_feature & false;
                                }
                            }
                        }
                    }

                    return return_feature;

                } else {
                    // If no filters are provided, return true for everything.
                    return return_feature;
                }
            }

            ,
        function4: function(feature, latlng, context) {
            const p = feature.properties;
            if (p.type === 'marker') {
                return L.marker(latlng);
            } else {
                return true;
            }
        }

    }
});