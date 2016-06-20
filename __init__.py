# -*- coding: utf-8 -*-
"""
/***************************************************************************
 LeastCostPath
                                 A QGIS plugin
 This Plugin used to find the least-cost path between to location on DEM Surface using anisotropic accumulated-cost surface and A* algorithm.
                             -------------------
        begin                : 2015-11-15
        copyright            : (C) 2015 by Achmad Faizal P S
        email                : faizalprbw@Gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load LeastCostPath class from file LeastCostPath.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .LeastCostPath import LeastCostPath
    return LeastCostPath(iface)
